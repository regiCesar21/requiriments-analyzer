"""
Classificador de requisitos — Opção C (híbrido):
  - F / NF  → modelo ML treinado no PURE (se disponível)
  - Regra de Negócio → regex (PURE não cobre essa categoria)
  - Fallback completo para regex se o modelo ainda não foi treinado
"""

import re
import joblib
from pathlib import Path

_base_path = Path(__file__).resolve().parents[2]
_MODEL_TYPE = 'model_train_requirements'

_MODEL_PRIORITY = ['bilstm', 'svm', 'logistic_regression', 'random_forest', 'naive_bayes']

# ── Cache de modelo ML ──────────────────────────────────────────────────────

_model_cache: dict = {}

# ── Cache do filtro semântico ───────────────────────────────────────────────

_semantic_cache: dict = {}

SEMANTIC_THRESHOLD         = 0.455  # threshold padrão (sem verbo de obrigação)
SEMANTIC_THRESHOLD_OBLIGED = 0.35   # threshold reduzido quando já há verbo de obrigação
CONFIDENCE_THRESHOLD       = 0.65   # probabilidade mínima do ML para não marcar como incerto

# Zero-Shot: usa NLI para classificar sem treino supervisionado.
# False → usa SVM (modo padrão); True → usa Zero-Shot (Hugging Face)
USE_ZERO_SHOT = False

# Cache do pipeline Zero-Shot (lazy)
_zs_cache: dict = {}

# Padrões de atribuição narrativa: "Ana disse que [requisito]", "ficou decidido que [req]"
_ATTRIBUTION_RE = re.compile(
    r'^.{0,80}\b(disse|anotou|pediu|lembrou|lembrar|concordou|estimou|levantou|registrou|'
    r'destacou|argumentou|complementou|mencionou|explicou|sugeriu|propôs|finalizou|'
    r'ressaltou|acrescentou|enfatizou|afirmou|reforçou|pontuou|'
    r'said|noted|mentioned|suggested|argued|stated|pointed out|emphasized|concluded|'
    r'ficou\s+(decidido|acordado|definido|estabelecido)|'
    r'foi\s+(decidido|acordado|definido|estabelecido)|'
    r'a\s+equipe\s+(concordou|decidiu|definiu)|'
    r'o\s+time\s+(concordou|decidiu|definiu))'
    r'(\s+\w+ndo)?'
    r'(\s+que\s+|\s+uma\s+(?:questão|dúvida|ponto|problema)\b[^:]{0,80}:\s*)',
    re.IGNORECASE,
)

_DISCOURSE_CONNECTOR_RE = re.compile(
    r'^(?:além\s+disso|adicionalmente|por\s+(?:fim|último)|também|ainda\s+assim|'
    r'in\s+addition|furthermore|additionally|finally|also|moreover)[,;]?\s+',
    re.IGNORECASE,
)

# Verbos de obrigação — indicam alta probabilidade de ser requisito
_OBLIGATION_RE = re.compile(
    r'\b(deve|devem|deverá|deverão|terá\s+que|têm\s+que|tem\s+que|precisa|precisam|'
    r'precisará|não\s+pode|não\s+podem|obrigatório|shall|must|will\s+be|should|'
    r'needs?\s+to|has\s+to|have\s+to|is\s+required)\b',
    re.IGNORECASE,
)


def _get_semantic_filter():
    """Carrega (lazy) o Sentence Transformer e os embeddings das âncoras PROMISE."""
    if 'loaded' in _semantic_cache:
        return _semantic_cache.get('model'), _semantic_cache.get('anchor_embs')

    _semantic_cache['loaded'] = True
    try:
        import torch
        from sentence_transformers import SentenceTransformer, util as st_util
        import pandas as pd

        csv_path = _base_path / 'data' / 'pure_requirements.csv'
        if not csv_path.exists():
            _semantic_cache['model'] = None
            return None, None

        df = pd.read_csv(csv_path)
        # Usa apenas âncoras em inglês para melhor qualidade do embedding
        anchors = df[df['text'].str.match(r'^[A-Za-z]')]['text'].tolist()
        if not anchors:
            anchors = df['text'].tolist()

        model = SentenceTransformer('all-MiniLM-L6-v2')
        anchor_embs = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False)

        _semantic_cache['model']       = model
        _semantic_cache['anchor_embs'] = anchor_embs
        print(f"[requirements_extractor] Filtro semântico carregado ({len(anchors)} âncoras)")
    except Exception as e:
        print(f"[requirements_extractor] Filtro semântico indisponível: {e}")
        _semantic_cache['model']       = None
        _semantic_cache['anchor_embs'] = None

    return _semantic_cache.get('model'), _semantic_cache.get('anchor_embs')


def _extract_core_clause(sentence: str) -> str:
    """Remove prefixo de atribuição narrativa e retorna a cláusula subordinada.

    Ex: "Ana anotou que o portal precisará mostrar..."
     → "o portal precisará mostrar..."
    Se não houver prefixo de atribuição, retorna a sentença original.
    """
    m = _ATTRIBUTION_RE.search(sentence)
    if m:
        return sentence[m.end():].strip()
    dc = _DISCOURSE_CONNECTOR_RE.match(sentence)
    return sentence[dc.end():].strip() if dc else sentence


def is_requirement_candidate(sentence: str, threshold: float = SEMANTIC_THRESHOLD) -> bool:
    """Retorna True se a sentença tem similaridade semântica suficiente com
    requisitos conhecidos (PROMISE NFR) para ser considerada candidata.

    Melhorias aplicadas:
    - Extrai cláusula subordinada antes de pontuar (remove "Ana disse que...")
    - Threshold adaptativo: 0.35 quando já há verbo de obrigação explícito
    """
    sem_model, anchor_embs = _get_semantic_filter()
    if sem_model is None or anchor_embs is None:
        return True

    try:
        import torch
        from sentence_transformers import util as st_util

        # Melhoria 2: threshold menor se a sentença já contém verbo de obrigação
        effective_threshold = threshold
        if _OBLIGATION_RE.search(sentence):
            effective_threshold = SEMANTIC_THRESHOLD_OBLIGED

        # Melhoria 1: pontua a cláusula core, não o prefixo narrativo
        core = _extract_core_clause(sentence)
        text_en = _translate_to_en(core) if _detect_lang(core) == 'pt' else core

        emb   = sem_model.encode([text_en], convert_to_tensor=True)
        sims  = st_util.cos_sim(emb, anchor_embs)[0]
        score = torch.topk(sims, k=min(10, len(sims))).values.mean().item()
        return score >= effective_threshold
    except Exception:
        return True


def _get_ml_model():
    """Carrega o melhor modelo disponível (lazy, cached no processo)."""
    if 'loaded' in _model_cache:
        return _model_cache.get('model')

    _model_cache['loaded'] = True
    _model_cache['model']  = None

    model_root = _base_path / 'model_train' / _MODEL_TYPE
    if not model_root.exists():
        return None

    versions = sorted(
        [d for d in model_root.iterdir() if d.is_dir() and d.name.startswith('version')],
        key=lambda p: int(p.name.replace('version', '')),
    )
    if not versions:
        return None

    latest = versions[-1]

    # Prefer best_model.txt recorded by the training script
    best_txt = latest / 'best_model.txt'
    priority = _MODEL_PRIORITY[:]
    if best_txt.exists():
        best_name = best_txt.read_text().strip()
        priority = [best_name] + [n for n in priority if n != best_name]

    for name in priority:
        path = latest / f'{name}_requirements_model.pkl'
        if path.exists():
            try:
                _model_cache['model'] = joblib.load(path)
                print(f"[requirements_extractor] Modelo ML carregado: {path.name}")
            except Exception as e:
                print(f"[requirements_extractor] Erro ao carregar modelo: {e}")
            break

    return _model_cache.get('model')


def reload_model():
    """Força recarregamento do modelo (chamar após novo treino)."""
    _model_cache.clear()


# ── Padrões regex (fallback e Regras de Negócio) ────────────────────────────

_FUNCTIONAL = [
    r'\bdeve\b', r'\bdeverá\b', r'\bprecisa\b', r'\bpermitir\b', r'\bpossibilitar\b',
    r'\bpermita\b', r'\bdeve realizar\b', r'\bdeve executar\b', r'\bo sistema\b',
    r'\bdeve ser capaz\b', r'\bshall\b', r'\bmust\b', r'\bshould\b',
    r'\bthe system\b', r'\bhas to\b', r'\bwill allow\b', r'\bmust provide\b',
    r'\bshall be able\b', r'\bneed to\b', r'\bresponsável por\b', r'\bresponsible for\b',
]

_NON_FUNCTIONAL = [
    r'\bdesempenho\b', r'\bperformance\b', r'\bdisponibilidade\b', r'\bavailability\b',
    r'\btempo de resposta\b', r'\bresponse time\b', r'\bsegurança\b', r'\bsecurity\b',
    r'\bescalabilidade\b', r'\bscalability\b', r'\busabilidade\b', r'\busability\b',
    r'\bconfiabilidade\b', r'\breliability\b', r'\blatência\b', r'\blatency\b',
    r'\bthroughput\b', r'\bacessibilidade\b', r'\baccessibility\b',
    r'\bmanutenibilidade\b', r'\bmaintainability\b', r'\beficiência\b', r'\befficiency\b',
    r'\bcapacidade\b', r'\bcapacity\b', r'\bportabilidade\b', r'\bportability\b',
    r'\bsla\b', r'\buptime\b', r'\btolerância a falha\b', r'\bfault tolerance\b',
    r'\bcriptograf', r'\bencriptad', r'\bencrypt', r'\baudit',
    r'\btransações por segundo\b', r'\brequests per second\b',
    r'\brespond within\b', r'\bwithin \d+ second', r'\bno máximo \d',
    r'\b99[.,]\d+\s*%', r'\bseconds?\s+under\b', r'\bunder.*load\b',
    r'\bcomply with\b', r'\bcumprimento\b', r'\bcompliance\b', r'\bconformidade\b',
]

_BUSINESS_RULE = [
    r'\bse\b.{3,80}\bentão\b', r'\bif\b.{3,80}\bthen\b',
    # se X, Y deve/tem que/será/precisa (vírgula em vez de "então")
    r'\bse\b.{3,80},.{3,60}\b(deve|tem que|terá que|será|precisa|deverá)\b',
    r'\bif\b.{3,80},.{3,60}\b(must|shall|will|should|needs to)\b',
    r'\bquando\b.{3,60}\bdeve\b', r'\bwhen\b.{3,60}\bmust\b',
    r'\bsomente se\b', r'\bapenas se\b', r'\bonly if\b', r'\bonly when\b',
    r'\bsomente [^s]', r'\bapenas [^s]', r'\bonly [^iw]',
    r'\bobrigatório\b', r'\bnão é permitido\b', r'\bnot allowed\b',
    r'\bnão pode\b', r'\bmust not\b', r'\bé necessário que\b', r'\bis required\b',
    r'\btoda vez que\b', r'\bwhenever\b', r'\bproibido\b', r'\bforbidden\b',
    r'\bcaso contrário\b', r'\botherwise\b', r'\bexceção\b', r'\bexception\b',
    # restrição de permissão: "só pode ser feito por X"
    r'\bsó\s+pode[m]?\b',
    # consequência de limite/threshold: "acima disso o sistema deverá"
    r'\bacima\s+(disso|desse\s+valor|do\s+limite|de\s+[R$])',
    r'\b(valor|limite|montante)\b.{3,60}\b(acima|superior|ultrapassar|exceder)\b',
    r'\b(above|exceeds?|over)\b.{3,60}\b(must|shall|will|should)\b',
    # ── BRs implícitas (sem marcadores se/então/only) ──────────────────────
    # Aprovação/autorização obrigatória
    r'\b(exige|requer)\s+(aprovação|autorização|confirmação|validação)\b',
    r'\b(requires?|needs?)\s+(approval|authorization|confirmation)\b',
    r'\bmediante\s+(aprovação|autorização|confirmação)\b',
    # Unicidade e restrição de dados
    r'\bdeve\s+ser\s+(único|exclusivo|distinto)\b',
    r'\bmust\s+be\s+(unique|distinct|exclusive)\b',
    # Threshold numérico genérico: "acima de 30%", "acima de R$500"
    r'\bacima\s+de\s+[R$]?\d',
    # Condicional "em caso de X, deve Y"
    r'\b(em\s+caso\s+de|caso\s+haja)\b.{3,80}\b(deve|deverá|precisa|será)\b',
]

# Conjunto de padrões BR compilado para checagem rápida pré-filtro
_BR_PRECHECK = re.compile(
    '|'.join(_BUSINESS_RULE),
    re.IGNORECASE,
)

# Sentenças com linguagem de desejo/usabilidade que passam pelo classificador
# mesmo sem verbo de obrigação formal (sem "deve/precisa/shall").
_NF_PRECHECK = re.compile(
    r'\bseja[m]?\s+(?:clara|amigáv|intuitiv|acessív|simpl|objetiv|fácil|legív)\w*\b'
    r'|\bquer(?:em)?\s+(?:poder|conseguir)\b'
    r'|\bdesejam?\s+(?:poder|conseguir)\b'
    r'|\bgostariam?\s+de\b',
    re.IGNORECASE,
)

# Cabeçalhos de documento e metadados de reunião/email — nunca são requisitos
_DOCUMENT_HEADER_RE = re.compile(
    r'\b(ata\s+de\s+reuni|projeto\s*:\s*\w|data\s*:\s*\d|participantes\s*:|'
    r'versão\s*:\s*\d|email\s+complementar|abraço,?\s*\w)',
    re.IGNORECASE,
)

# Contexto organizacional ou de projeto — sujeito não é o sistema de software
_ORGANIZATIONAL_RE = re.compile(
    r'\b(banco|empresa|organização|instituição|corporação|clínica|hospital|'
    r'escola|loja|startup|companhia|cooperativa)\b.{3,100}'
    r'\b(precisa|quer|vai|está|deseja)\s+(modernizar|digitalizar|atualizar|renovar|'
    r'substituir|transformar|evoluir|expandir|melhorar\s+o\s+canal|lançar|implementar)\b'
    r'|\btime\s+de\s+(?:qa|qualidade|test\w*|dev\w*)\b.{0,80}'
    r'\b(precisará?|vai|quer|deverá)\s+(?:de\s+)?(?:acesso|ambiente)\b'
    # Objetivo organizacional: "objetivo principal é lançar uma plataforma"
    r'|\b(objetivo\s+(?:principal|geral|estratégico|do\s+projeto)|meta\s+(?:principal|geral))\b'
    r'.{0,100}\b(lançar|criar|desenvolver|implementar|disponibilizar|construir|migrar)\b'
    # Decisão de escopo MVP: "o MVP contemplará apenas agendamento"
    r'|\bo\s+mvp\b.{0,80}\b(contempla|contemplará|inclui|incluirá|apenas|somente|sem\s+módulo)\b',
    re.IGNORECASE,
)

# Sujeito de sistema explícito — presença garante que NÃO é contexto organizacional
_SYSTEM_SUBJECT_RE = re.compile(
    r'\b(o\s+sistema|a\s+aplicação|o\s+portal|a\s+plataforma|o\s+módulo|'
    r'o\s+app\b|o\s+software|the\s+system|the\s+application|the\s+platform|'
    r'the\s+module|the\s+service)\b',
    re.IGNORECASE,
)


def _is_org_context_ner(sentence: str) -> bool:
    """Detecta via NER (SpaCy) se o sujeito é uma organização/pessoa nomeada —
    indicando contexto organizacional, não requisito do sistema de software.

    Aplica apenas quando:
    - Não há sujeito de sistema explícito (o sistema / o portal / the application)
    - Há verbo de obrigação (candidato a requisito)
    """
    if _SYSTEM_SUBJECT_RE.search(sentence):
        return False
    if not _OBLIGATION_RE.search(sentence):
        return False

    if 'spacy_nlp' not in _semantic_cache:
        try:
            import spacy
            _semantic_cache['spacy_nlp'] = spacy.load('en_core_web_sm')
        except Exception:
            _semantic_cache['spacy_nlp'] = None

    nlp = _semantic_cache.get('spacy_nlp')
    if nlp is None:
        return False

    try:
        core = _extract_core_clause(sentence)
        text_en = _translate_to_en(core) if _detect_lang(core) == 'pt' else core
        doc = nlp(text_en)
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'GPE') and ent.start <= 8:
                return True
    except Exception:
        pass
    return False

# ── Camada de desambiguação Funcional ↔ Não-Funcional ──────────────────────
#
# O SVM treinado no PROMISE NFR (requisitos isolados e estruturados) tende a
# classificar como NF qualquer sentença com verbos de obrigação em texto corrido,
# pois não aprendeu a distinção semântica entre "executar funcionalidade" e
# "satisfazer atributo de qualidade".
#
# Estratégia: após o SVM, verificar sinais de domínio para corrigir casos
# onde o modelo erra sistematicamente.

# Sinais fortes de NF: métricas quantitativas, conformidade, qualidade de sistema
_NF_STRONG = re.compile(
    # Bug fix: transaç\b não casa "transações" — usar prefixo sem \b final
    r'\d+\s*(segundo|ms|milissegundo|minuto|hora|%|MB|GB|KB|req|transaç\w*|usuário|'
    r'second|minute|hour|request|user|tps|rps)\b'
    r'|'
    r'\b(lgpd|wcag|iso\s*\d+|pci|gdpr|soc\s*2|bacen|banco\s*central|'
    r'conformidade|compliance|auditoria\w*|audit\w*|log\s+de\s+audit|'
    r'criptograf|encriptad|encrypt|tls|ssl|https|bcrypt|argon|hash\b|salt\b|'
    r'disponibilidade|uptime|sla\b|tolerância\s+a\s+falha|fault\s+tolerance|'
    r'tempo\s+de\s+resposta|response\s+time|latên|throughput|escalab|'
    r'acessibilidade.*deficiên|screen.?read)\b'
    # Usabilidade: "seja clara e amigável", "sem jargão", "user-friendly"
    r'|\bseja[m]?\s+(?:clara|amigáv|intuitiv|acessív|simpl|objetiv|fácil|legív)\w*\b'
    r'|\bsem\s+jargão\b|\buser.?friendly\b|\bfácil\s+de\s+(?:usar|navegar|entender)\b'
    # Restrição de UX: "não pode ter mais de X telas", "máximo de X etapas"
    r'|\b(?:no\s+máximo|máximo\s+de|não\s+pode[m]?\s+ter\s+mais\s+de)\s+\d+\s*'
    r'(?:tela|passo|etapa|clique|pant|step|screen|click|campo)\w*\b',
    re.IGNORECASE,
)

# Sinais fortes de Funcional: verbos de ação transitivos sobre dados/interface
# (distinguem "o sistema deve MOSTRAR X" de "o sistema deve SER rápido")
_FUNCTIONAL_ACTION = re.compile(
    # Permite verbos modais intermediários: "precisam conseguir fazer", "deve poder enviar"
    r'\b(deve|devem|deverá|precisa|precisam|precisará|tem que|têm que|'
    r'quer(?:em)?\s+(?:poder|conseguir)|desejam?\s+(?:poder|conseguir)|'
    r'shall|must|will|should|needs?\s+to)'
    r'(\s+(?:conseguir|poder|ser\s+capaz\s+de|be\s+able\s+to))?'
    r'\s+(?:se\s+)?'   # pronome reflexivo opcional: "deve se integrar"
    r'(mostrar|exibir|apresentar|visualizar|'
    r'enviar|notificar|alertar|disparar|'
    r'permitir|possibilitar|habilitar|disponibilizar|'
    r'fazer|realizar|executar|processar|'
    r'configurar|personalizar|definir|ajustar|'
    r'acessar|consultar|buscar|pesquisar|filtrar|listar|'
    r'cadastrar|registrar|salvar|armazenar(?!\s+senha)|'  # armazenar senha é NF/BR
    r'cancelar|suspender|encerrar|fechar|'
    r'gerar|criar|produzir|emitir|exportar|baixar|importar|'
    r'integrar|conectar|comunicar|sincronizar|'
    r'autenticar(?!\s+e\s+autorizar)|logar|fazer\s+login|'
    r'display|show|send|allow|enable|provide|create|generate|'
    r'list|search|filter|view|access|download|upload|export|import)\b',
    re.IGNORECASE,
)

# Proibições de qualidade (não é BR, é requisito de segurança/integridade)
# Ex: "não pode armazenar senhas em texto plano" → NF, não BR
_NF_PROHIBITION = re.compile(
    r'\b(não\s+pode[m]?|must\s+not|shall\s+not)\s+'
    r'(armazenar\s+senhas?|guardar\s+senhas?|salvar\s+senhas?|store\s+passwords?|'
    r'expor|vazar|leak|transmitir\s+em\s+claro|transmit.*plain|'
    # Restrição de UX: "não pode ter mais de 3 telas/etapas/passos"
    r'ter\s+mais\s+de\s+\d+)',
    re.IGNORECASE,
)


def _disambiguate_fn_nf(sentence: str, svm_pred: str) -> str:
    """Aplica sinais de domínio para corrigir confusão Funcional ↔ NF do SVM.

    Regras aplicadas (em ordem de prioridade):
    1. Proibição de qualidade (não pode armazenar senha) → NF, não BR
    2. Sinal forte de NF (métrica, conformidade, segurança) → NF
    3. Sinal forte de Funcional (verbo de ação + sem NF forte) → Functional
    4. Caso contrário → mantém predição do SVM
    """
    # Regra 1: proibição de qualidade nunca é BR
    if svm_pred == 'business_rule' and _NF_PROHIBITION.search(sentence):
        return 'non_functional'

    # Regra 2: sinal forte de NF tem prioridade sobre Funcional
    if _NF_STRONG.search(sentence):
        if svm_pred in ('functional', 'uncertain'):
            return 'non_functional'
        return svm_pred  # NF ou BR já correto

    # Regra 3: verbo de ação transitivo sem sinal NF → provavelmente Funcional
    if _FUNCTIONAL_ACTION.search(sentence):
        if svm_pred in ('non_functional', 'uncertain'):
            return 'functional'

    return svm_pred


TYPE_LABELS = {
    'functional':     'Funcional',
    'non_functional': 'Não-Funcional',
    'business_rule':  'Regra de Negócio',
    'uncertain':      'Incerto',
}

TYPE_ICONS = {
    'functional':     '🟢',
    'non_functional': '🟡',
    'business_rule':  '🔵',
    'uncertain':      '⚪',
}


# ── Detecção de idioma e tradução ───────────────────────────────────────────

def _detect_lang(text: str) -> str:
    """Retorna 'pt', 'en' ou outro código ISO."""
    try:
        import langid
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return 'en'


def _translate_to_en(text: str) -> str:
    """Traduz texto PT→EN para uso interno no classificador ML."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='pt', target='en').translate(text) or text
    except Exception:
        return text


# ── Classificação ───────────────────────────────────────────────────────────

def classify_sentence(sentence: str, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """Classifica usando ML (F/NF) + regex (Regra de Negócio).

    Retorna 'functional', 'non_functional', 'business_rule', 'uncertain' ou 'irrelevant'.
    - Para texto em PT: traduz para EN antes de passar pelo modelo ML.
    - Se a confiança do modelo for menor que `threshold`, retorna 'uncertain'.
    """
    if not sentence or len(sentence.split()) < 5:
        return 'irrelevant'

    text = sentence.lower()

    # Regras de Negócio: regex funciona em PT e EN nativamente
    br_score = sum(1 for p in _BUSINESS_RULE if re.search(p, text))

    model = _get_ml_model()
    if model is not None:
        # Detecta idioma e traduz PT→EN antes do ML
        lang = _detect_lang(sentence)
        text_for_ml = _translate_to_en(sentence) if lang == 'pt' else sentence

        # BR via regex tem prioridade sobre ML —
        # exceto proibições de qualidade ("não pode armazenar senhas") que são NF
        if br_score >= 1 and not _NF_PROHIBITION.search(sentence):
            return 'business_rule'

        # Threshold de confiança — usa predict_proba se disponível
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text_for_ml])[0]
            confidence = proba.max()
            # Bug fix: passar 'uncertain' pela desambiguação — sinais de domínio
            # fortes (LGPD, auditoria, métricas) devem sobrescrever baixa confiança
            if confidence < threshold:
                return _disambiguate_fn_nf(sentence, 'uncertain')
            ml_pred = model.classes_[proba.argmax()]
        else:
            ml_pred = model.predict([text_for_ml])[0]

        return _disambiguate_fn_nf(sentence, ml_pred)

    # ── Fallback regex (modelo ainda não treinado) ──────────────────────────
    fn = sum(1 for p in _FUNCTIONAL     if re.search(p, text))
    nf = sum(1 for p in _NON_FUNCTIONAL if re.search(p, text))

    if fn == 0 and nf == 0 and br_score == 0:
        return 'irrelevant'
    # NF keywords are domain-specific → take priority over generic functional verbs
    if nf > 0:
        return 'non_functional'
    if br_score >= 1:
        return 'business_rule'
    return 'functional'


def using_ml_model() -> bool:
    """Retorna True se o modelo ML está carregado (útil para exibir na UI)."""
    return _get_ml_model() is not None


# ── Zero-Shot Classification (Hugging Face NLI) ─────────────────────────────

# Hipóteses em inglês — modelo NLI entende EN melhor que PT
_ZS_LABELS = [
    "the system must perform an action or provide a feature to the user",
    "quality attribute such as performance, security, reliability, availability, compliance or accessibility",
    "conditional business rule: if condition then obligation, or explicit prohibition or restriction",
]

# Mapeamento do label retornado pelo modelo → tipo interno
_ZS_LABEL_MAP = {
    "the system must perform an action or provide a feature to the user":                                      "functional",
    "quality attribute such as performance, security, reliability, availability, compliance or accessibility": "non_functional",
    "conditional business rule: if condition then obligation, or explicit prohibition or restriction":          "business_rule",
}

_ZS_CONFIDENCE_THRESHOLD = 0.40  # score mínimo do label vencedor


def _get_zs_pipeline():
    """Carrega (lazy) o pipeline de Zero-Shot Classification."""
    if 'loaded' in _zs_cache:
        return _zs_cache.get('pipe')

    _zs_cache['loaded'] = True
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )
        _zs_cache['pipe'] = pipe
        print("[requirements_extractor] Zero-Shot pipeline carregado (DeBERTa NLI)")
    except Exception as e:
        print(f"[requirements_extractor] Zero-Shot indisponível: {e}")
        _zs_cache['pipe'] = None

    return _zs_cache.get('pipe')


def classify_sentence_zeroshot(sentence: str) -> str:
    """Classifica usando Zero-Shot Classification (NLI via Hugging Face).

    Traduz PT→EN antes de classificar para aproveitar melhor o modelo NLI.
    Retorna 'functional', 'non_functional', 'business_rule' ou 'uncertain'.
    """
    pipe = _get_zs_pipeline()
    if pipe is None:
        return classify_sentence(sentence)  # fallback para SVM

    text = sentence.lower()

    # Regras de Negócio via regex têm prioridade (são determinísticas)
    br_score = sum(1 for p in _BUSINESS_RULE if re.search(p, text))
    if br_score >= 1:
        return 'business_rule'

    try:
        lang = _detect_lang(sentence)
        text_en = _translate_to_en(sentence) if lang == 'pt' else sentence

        result = pipe(text_en, candidate_labels=_ZS_LABELS, multi_label=False)
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        if top_score < _ZS_CONFIDENCE_THRESHOLD:
            return 'uncertain'

        zs_pred = _ZS_LABEL_MAP.get(top_label, 'uncertain')
        return _disambiguate_fn_nf(sentence, zs_pred)
    except Exception:
        return 'uncertain'


def extract_requirements(sentences: list[str], use_zero_shot: bool = None) -> list[dict]:
    """Classifica sentenças e retorna apenas as identificadas como requisitos.

    Pipeline de dois estágios:
    1. Filtro semântico — descarta sentenças sem similaridade com requisitos PROMISE
    2. Classificador (SVM ou Zero-Shot) — decide o tipo (F/NF/BR/uncertain)

    Parâmetro `use_zero_shot`: None → usa a flag global USE_ZERO_SHOT.
    """
    from helpers.requirements_analyzer import score_requirement

    zs = USE_ZERO_SHOT if use_zero_shot is None else use_zero_shot
    classify_fn = classify_sentence_zeroshot if zs else classify_sentence

    results = []
    for s in sentences:
        # Rejeitar cabeçalhos e contexto organizacional por padrões conhecidos
        if _DOCUMENT_HEADER_RE.search(s) or _ORGANIZATIONAL_RE.search(s):
            continue
        # Sentenças com marcador BR ou NF explícito pulam o filtro semântico
        has_br_marker = bool(_BR_PRECHECK.search(s.lower()))
        has_nf_marker = bool(_NF_PRECHECK.search(s))
        if not has_br_marker and not has_nf_marker and not is_requirement_candidate(s):
            continue
        # NER: detecta contexto organizacional não capturado por regex
        if _is_org_context_ner(s):
            continue
        req_type = classify_fn(s)
        if req_type != 'irrelevant':
            quality = score_requirement(s)
            results.append({
                'text':           s,
                'type':           req_type,
                'quality_score':  quality['score'],
                'quality_label':  quality['label'],
                'quality_icon':   quality['icon'],
                'quality_issues': quality['issues'],
            })
    return results
