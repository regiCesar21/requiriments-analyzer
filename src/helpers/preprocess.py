from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

def load_spacy_model():
    import spacy
    return spacy.load('en_core_web_sm')

def preprocess_with_spacy(texts, nlp):
    processed_texts = []
    for doc in texts:
        spacy_doc = nlp(doc.lower())
        tokens = [
            token.lemma_
            for token in spacy_doc
            if not token.is_stop and not token.is_punct
        ]
        processed_texts.append(tokens)
    return processed_texts


# ── Sklearn transformer reutilizável ────────────────────────────────────────

_nlp_cache: dict = {}

def _load_nlp(lang: str):
    import spacy
    if lang not in _nlp_cache:
        model = 'en_core_web_sm' if lang == 'en' else 'pt_core_news_lg'
        _nlp_cache[lang] = spacy.load(model, disable=['parser', 'ner'])
    return _nlp_cache[lang]


class SpacyLemmaTokenizer(BaseEstimator, TransformerMixin):
    """Sklearn transformer: lowercase → lematização → remoção de stops/punct.

    Compatível com joblib.dump/load: o modelo SpaCy é carregado lazily
    e excluído da serialização via __getstate__.
    """

    def __init__(self, lang: str = 'en', max_len: int = 512):
        self.lang = lang
        self.max_len = max_len
        self._nlp = None

    # Exclui o modelo SpaCy (não-serializável) do pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_nlp'] = None
        return state

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = _load_nlp(self.lang)
        return self._nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = self._get_nlp()
        result = []
        for text in X:
            doc = nlp(str(text).lower()[:self.max_len])
            tokens = [
                t.lemma_ for t in doc
                if not t.is_stop and not t.is_punct and len(t.text.strip()) > 1
            ]
            result.append(' '.join(tokens) if tokens else str(text).lower())
        return result


# ── BiLSTM ──────────────────────────────────────────────────────────────────

def build_vocab(texts: list, min_freq: int = 1) -> dict:
    """Constrói vocabulário word→index a partir de uma lista de textos."""
    counter = Counter(w for t in texts for w in t.lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict, max_len: int) -> list:
    """Converte texto em sequência de inteiros com padding/truncamento."""
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (max_len - len(ids))
    return ids


class _BiLSTMModel:
    """Wrapper lazy para evitar import de torch no topo do módulo."""

    def __new__(cls, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True,
                                    batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim * 2, num_classes)

            def forward(self, x):
                emb = self.dropout(self.embedding(x))
                _, (hidden, _) = self.lstm(emb)
                out = self.dropout(
                    __import__('torch').cat([hidden[0], hidden[1]], dim=1)
                )
                return self.fc(out)

        return _Net()


class BiLSTMWrapper:
    """Interface sklearn-compatível (.predict) para o modelo BiLSTM PyTorch.

    Serialização: o state_dict do modelo é salvo em vez do objeto nn.Module,
    permitindo joblib.dump/load sem dependências de device ou arquitetura.
    """

    def __init__(self, model, vocab: dict, label_map: dict, max_len: int):
        self.model = model
        self.vocab = vocab
        self.label_map = label_map
        self.max_len = max_len

    def predict(self, texts: list) -> list:
        import torch
        self.model.eval()
        results = []
        for text in texts:
            ids = encode_text(text, self.vocab, self.max_len)
            tensor = torch.LongTensor([ids])
            with torch.no_grad():
                pred = self.model(tensor).argmax(dim=1).item()
            results.append(self.label_map[pred])
        return results

    def __getstate__(self):
        state = self.__dict__.copy()
        m = state['model']
        state['_state_dict'] = m.state_dict()
        state['_arch'] = {
            'vocab_size': m.embedding.num_embeddings,
            'embed_dim':  m.embedding.embedding_dim,
            'hidden_dim': m.lstm.hidden_size,
            'num_classes': m.fc.out_features,
        }
        state['model'] = None
        return state

    def __setstate__(self, state):
        arch = state.pop('_arch')
        sd   = state.pop('_state_dict')
        self.__dict__.update(state)
        self.model = _BiLSTMModel(**arch)
        self.model.load_state_dict(sd)
        self.model.eval()