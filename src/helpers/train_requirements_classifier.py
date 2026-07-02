"""
Etapa 1 + 2 da Opção C:
  - Baixa o dataset PURE/PROMISE do HuggingFace (nickmuchi/requirements-classification)
  - Normaliza labels para 'functional' / 'non_functional'
  - Treina Naive Bayes, Logistic Regression e Random Forest com TF-IDF (1-2 grams)
  - Salva os .pkl em model_train/model_train_requirements/version<N>/
  - Salva CSV do dataset em data/pure_requirements.csv
"""

from pathlib import Path
from os.path import join

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
from helpers.preprocess import SpacyLemmaTokenizer, build_vocab, encode_text, _BiLSTMModel, BiLSTMWrapper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

base_path = Path(__file__).resolve().parents[2]

# ── Mapeamento de labels ────────────────────────────────────────────────────

# Labels funcionais nas convenções PROMISE / nickmuchi
_F_STRINGS = {'F', 'f', 'functional', 'Functional', '1'}

# Todas as demais (subcategorias de NF)
_NF_STRINGS = {
    'A', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US',
    'a', 'ft', 'l', 'lf', 'mn', 'o', 'pe', 'sc', 'se', 'us',
    'NF', 'nf', 'non-functional', 'nonfunctional', 'Non-Functional',
    '0',
}

# Mapeamento numérico PROMISE (ordem alfabética das classes):
# 0=A, 1=F, 2=FT, 3=L, 4=LF, 5=MN, 6=O, 7=PE, 8=SC, 9=SE, 10=US
_PROMISE_NUMERIC_F = {1}


def _normalize_label(raw) -> str:
    s = str(raw).strip()
    if s in _F_STRINGS:
        return 'functional'
    if s in _NF_STRINGS:
        return 'non_functional'
    try:
        n = int(float(s))
        return 'functional' if n in _PROMISE_NUMERIC_F else 'non_functional'
    except ValueError:
        return 'non_functional'


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    text_candidates  = ['RequirementText', 'text', 'sentence', 'requirement', 'Text']
    label_candidates = ['class', 'label', 'Label', 'type', 'category', 'Class']

    text_col  = next((c for c in text_candidates  if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Colunas não detectadas. Encontradas: {list(df.columns)}\n"
            f"Esperado texto em {text_candidates}, label em {label_candidates}"
        )
    return text_col, label_col


# ── Etapa 1: Download ───────────────────────────────────────────────────────

def download_pure_dataset() -> pd.DataFrame:
    """Baixa PROMISE NFR dataset e retorna DataFrame com colunas text/label.

    Tenta em ordem:
    1. CSV direto do GitHub (aashgar/mldata — PROMISE NFR 969 req.)
    2. HuggingFace datasets (fallback)
    """
    import urllib.request, io

    # ── Fonte primária: CSV público do PROMISE NFR ──────────────────────────
    csv_url = (
        "https://raw.githubusercontent.com/aashgar/mldata/master/nfr_exp.csv"
    )
    try:
        print(f"[download] Baixando PROMISE NFR de {csv_url}...")
        with urllib.request.urlopen(csv_url, timeout=60) as r:
            content = r.read().decode('utf-8', errors='replace')
        df_raw = pd.read_csv(io.StringIO(content))
        text_col, label_col = _detect_columns(df_raw)
        df = pd.DataFrame({
            'text':  df_raw[text_col].astype(str),
            'label': df_raw[label_col].apply(_normalize_label),
        })
        df = df[df['text'].str.len() > 10].dropna()
        print(f"[download] Carregados {len(df)} exemplos do PROMISE NFR")
        print(df['label'].value_counts().to_string())
        return df
    except Exception as e:
        print(f"[download] Fonte primária falhou: {e}")

    # ── Fallback: HuggingFace datasets ─────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Instale 'datasets': pip install datasets")

    hf_candidates = [
        "MariaIsabel/FR_NFR_Spanish_requirements_classification",
    ]
    for name in hf_candidates:
        try:
            print(f"[download] Tentando HuggingFace '{name}'...")
            ds = load_dataset(name)
            all_splits = [ds[s].to_pandas() for s in ds.keys()]
            df_raw = pd.concat(all_splits, ignore_index=True)
            print(f"[download] Carregados {len(df_raw)} exemplos de '{name}'")
            text_col, label_col = _detect_columns(df_raw)
            df = pd.DataFrame({
                'text':  df_raw[text_col].astype(str),
                'label': df_raw[label_col].apply(_normalize_label),
            })
            df = df[df['text'].str.len() > 10].dropna()
            print(f"[download] Após limpeza: {len(df)} exemplos")
            print(df['label'].value_counts().to_string())
            return df
        except Exception as e:
            print(f"[download] Falhou ({name}): {e}")

    raise RuntimeError(
        "Nenhum dataset pôde ser carregado.\n"
        "Verifique a conexão com internet e tente novamente."
    )


# ── Helpers de pasta (mesmo padrão de classifier.py) ───────────────────────

def _next_version_dir(root: Path, prefix: str) -> Path:
    existing = [
        int(p.name.replace(prefix, ''))
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    ] if root.exists() else []
    nxt = max(existing, default=0) + 1
    d = root / f'{prefix}{nxt}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_confusion_matrix(y_test, preds, model_name: str, out_dir: Path):
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão — {model_name}")
    fname = model_name.replace(' ', '_').lower() + '_confusion_matrix.png'
    plt.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()


# ── Etapa 2: Treino ─────────────────────────────────────────────────────────

def _build_bilingual_dataset(df_en: pd.DataFrame) -> pd.DataFrame:
    """Traduz o dataset EN para PT e combina com o original.

    Retorna DataFrame bilíngue com ~2x mais exemplos.
    """
    from deep_translator import GoogleTranslator

    print(f"\n[bilíngue] Traduzindo {len(df_en)} requisitos EN→PT...")
    pt_texts = []
    for i, text in enumerate(df_en['text']):
        try:
            translated = GoogleTranslator(source='en', target='pt').translate(str(text))
            pt_texts.append(translated or text)
        except Exception:
            pt_texts.append(text)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df_en)} traduzidos...")

    df_pt = pd.DataFrame({'text': pt_texts, 'label': df_en['label'].values})
    df_bilingual = pd.concat([df_en, df_pt], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    print(f"[bilíngue] Dataset final: {len(df_bilingual)} exemplos (EN + PT)")
    print(df_bilingual['label'].value_counts().to_string())
    return df_bilingual


def train_requirements_classifier(df: pd.DataFrame = None, bilingual: bool = True):
    """Treina NB, LR, RF e BiLSTM no dataset PROMISE NFR.

    df: se None, baixa automaticamente.
    bilingual: se True, expande o dataset com versão PT traduzida (~2x mais dados).
    Salva modelos em model_train/model_train_requirements/version<N>/.
    """
    if df is None:
        df = download_pure_dataset()

    if bilingual:
        df = _build_bilingual_dataset(df)

    # Persiste CSV
    data_path = base_path / 'data'
    data_path.mkdir(exist_ok=True)
    csv_name  = 'bilingual_requirements.csv' if bilingual else 'pure_requirements.csv'
    csv_path  = data_path / csv_name
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[treino] Dataset salvo em: {csv_path}")

    X, y = df['text'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"[treino] Treino: {len(X_train)} | Teste: {len(X_test)}")

    type_train = 'model_train_requirements'
    version_dir = _next_version_dir(base_path / 'model_train' / type_train, 'version')
    mat_dir     = _next_version_dir(base_path / 'image'       / type_train, 'matrices')

    classifiers = [
        (MultinomialNB(),                                        "Naive Bayes"),
        (LogisticRegression(max_iter=1000),                      "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100),               "Random Forest"),
        (CalibratedClassifierCV(LinearSVC(max_iter=2000), cv=5), "SVM"),
    ]

    log_path = base_path / 'log' / 'classification_reports.txt'
    best = {'name': None, 'score': 0.0, 'path': None}

    for clf, name in classifiers:
        pipeline = Pipeline([
            ('lemma', SpacyLemmaTokenizer(lang='en')),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', clf),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc    = (preds == y_test).mean()
        report = classification_report(y_test, preds, digits=3)

        print(f"\n{'='*50}\n{name}\n{report}")

        fname      = name.replace(' ', '_').lower()
        model_path = version_dir / f'{fname}_requirements_model.pkl'
        joblib.dump(pipeline, model_path)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[requirements] {name}\n{report}\n{'='*60}\n")

        _save_confusion_matrix(y_test, preds, name, mat_dir)

        if acc > best['score']:
            best = {'name': name, 'score': acc, 'path': str(model_path)}

    # ── BiLSTM ──────────────────────────────────────────────────────────────
    bilstm_acc, bilstm_path = _train_bilstm(
        X_train, X_test, y_train, y_test, version_dir, mat_dir, log_path
    )
    if bilstm_acc > best['score']:
        best = {'name': 'BiLSTM', 'score': bilstm_acc, 'path': bilstm_path}

    # Persiste o nome do melhor modelo para o extrator usar
    fname_best = best['name'].replace(' ', '_').lower()
    (version_dir / 'best_model.txt').write_text(fname_best, encoding='utf-8')

    print(f"\n{'='*50}")
    print(f"Melhor modelo : {best['name']}")
    print(f"Accuracy      : {best['score']:.3f}")
    print(f"Modelos salvos: {version_dir}")
    return best


def _train_bilstm(X_train, X_test, y_train, y_test, out_dir, mat_dir, log_path):
    """Treina BiLSTM com PyTorch e salva como bilstm_requirements_model.pkl."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[BiLSTM] PyTorch não encontrado — pulando. (pip install torch)")
        return 0.0, None

    MAX_LEN    = 50
    EMBED_DIM  = 100
    HIDDEN_DIM = 128
    EPOCHS     = 30
    BATCH_SIZE = 32
    LR         = 1e-3
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Vocabulário e encoding
    vocab      = build_vocab(X_train.tolist())
    labels     = sorted(set(y_train))
    label2idx  = {l: i for i, l in enumerate(labels)}
    idx2label  = {i: l for l, i in label2idx.items()}

    X_tr = torch.LongTensor([encode_text(t, vocab, MAX_LEN) for t in X_train])
    X_te = torch.LongTensor([encode_text(t, vocab, MAX_LEN) for t in X_test])
    y_tr = torch.LongTensor([label2idx[l] for l in y_train])
    y_te = torch.LongTensor([label2idx[l] for l in y_test])

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model     = _BiLSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*50}\nBiLSTM")
    print(f"  vocab={len(vocab)} | embed={EMBED_DIM} | hidden={HIDDEN_DIM} | device={device}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_te.to(device)).argmax(dim=1).cpu()
            val_acc = (val_preds == y_te).float().mean().item()
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.3f}")

    # Avaliação final
    model.eval()
    with torch.no_grad():
        preds_idx = model(X_te.to(device)).argmax(dim=1).cpu().numpy()

    preds_labels  = [idx2label[i] for i in preds_idx]
    y_test_labels = list(y_test)
    acc    = sum(p == t for p, t in zip(preds_labels, y_test_labels)) / len(y_test_labels)
    report = classification_report(y_test_labels, preds_labels, digits=3)
    print(report)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[requirements] BiLSTM\n{report}\n{'='*60}\n")

    _save_confusion_matrix(y_test_labels, preds_labels, "BiLSTM", mat_dir)

    # Salva wrapper compatível com sklearn
    wrapper    = BiLSTMWrapper(model.cpu(), vocab, idx2label, MAX_LEN)
    model_path = out_dir / 'bilstm_requirements_model.pkl'
    joblib.dump(wrapper, model_path)

    return acc, str(model_path)
