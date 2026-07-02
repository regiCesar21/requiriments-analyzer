"""
Análise pós-classificação de requisitos:
  - score_requirement()   → qualidade individual (IEEE 830)
  - find_duplicates()     → pares semanticamente similares
  - group_requirements()  → agrupamento temático (K-means)
"""

import re

# ── Qualidade ────────────────────────────────────────────────────────────────

_VAGUE_WORDS = [
    'adequado', 'rápido', 'fácil', 'amigável', 'intuitivo', 'robusto',
    'eficiente', 'simples', 'flexível', 'moderno', 'geralmente', 'normalmente',
    'sufficient', 'fast', 'easy', 'friendly', 'intuitive', 'robust',
    'efficient', 'simple', 'flexible', 'modern', 'usually', 'generally',
    'appropriate', 'adequate', 'reasonable', 'suitable',
]

_ACTION_VERBS = r'\b(devem?|deverá|precisam?|tem que|têm que|shall|must|will|should|needs? to|has to)\b'
_MEASURABLE   = r'\d+\s*(segundo|ms|milissegundo|minuto|hora|%|MB|GB|KB|req|transaç|usuário|second|minute|hour|request|user)'
_COMPOUND_REQ = r'\b(deve|shall|must)\b.{5,80}\b(e|and)\b.{5,80}\b(deve|shall|must)\b'
_HAS_ACTOR    = r'\b(sistema|usuário|aplicação|serviço|módulo|system|user|application|service|module|cliente|client)\b'

_QUALITY_LABELS = {4: 'Excelente', 3: 'Bom', 2: 'Regular', 1: 'Ruim', 0: 'Inválido'}
_QUALITY_ICONS  = {4: '🟢', 3: '🟡', 2: '🟠', 1: '🔴', 0: '⛔'}


def score_requirement(text: str) -> dict:
    """Avalia a qualidade de um requisito com base em heurísticas IEEE 830.

    Retorna dict com score (0-4), label, ícone e lista de problemas encontrados.
    """
    t = text.lower()
    score  = 0
    issues = []

    # 1. Tem verbo de obrigação claro
    if re.search(_ACTION_VERBS, t):
        score += 1
    else:
        issues.append("Sem verbo de obrigação (deve, shall, must...)")

    # 2. É mensurável / testável
    if re.search(_MEASURABLE, t):
        score += 1
    else:
        issues.append("Sem critério mensurável (número, percentual, tempo...)")

    # 3. Não usa palavras vagas
    found_vague = [v for v in _VAGUE_WORDS if re.search(r'\b' + v + r'\b', t)]
    if not found_vague:
        score += 1
    else:
        issues.append(f"Palavras vagas: {', '.join(found_vague)}")

    # 4. Requisito único (não composto)
    if not re.search(_COMPOUND_REQ, t):
        score += 1
    else:
        issues.append("Possível requisito composto (múltiplos 'deve/shall')")

    return {
        'score':     score,
        'max_score': 4,
        'label':     _QUALITY_LABELS[score],
        'icon':      _QUALITY_ICONS[score],
        'issues':    issues,
    }


# ── Duplicatas ───────────────────────────────────────────────────────────────

def find_duplicates(requirements: list[dict], threshold: float = 0.82) -> list[dict]:
    """Detecta pares de requisitos semanticamente similares.

    Retorna lista de dicts com os dois requisitos e o score de similaridade.
    Usa Sentence Transformers (all-MiniLM-L6-v2).
    """
    if len(requirements) < 2:
        return []

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util

        model  = SentenceTransformer('all-MiniLM-L6-v2')
        texts  = [r['text'] for r in requirements]
        embs   = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.cos_sim(embs, embs)

        duplicates = []
        n = len(requirements)
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i][j].item()
                if score >= threshold:
                    duplicates.append({
                        'req_a':      requirements[i],
                        'req_b':      requirements[j],
                        'similarity': round(score, 3),
                    })

        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    except Exception as e:
        print(f"[requirements_analyzer] find_duplicates falhou: {e}")
        return []


# ── Agrupamento ──────────────────────────────────────────────────────────────

_GROUP_LABELS = [
    'Autenticação e Acesso',
    'Performance e Escalabilidade',
    'Segurança e Privacidade',
    'Interface e Usabilidade',
    'Integração e Dados',
    'Disponibilidade e Confiabilidade',
    'Regras de Negócio',
    'Outros',
]


def _auto_label_group(texts: list[str]) -> str:
    """Gera um rótulo automático para um grupo baseado nas palavras mais frequentes."""
    from collections import Counter
    stopwords = {
        'o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na',
        'para', 'por', 'com', 'que', 'se', 'the', 'a', 'an', 'of', 'to', 'in',
        'and', 'or', 'for', 'with', 'shall', 'must', 'deve', 'system', 'sistema',
        'user', 'usuário', 'be', 'able', 'allow', 'provide', 'support',
    }
    words = []
    for t in texts:
        words.extend(w.lower() for w in re.findall(r'\b[a-záéíóúãõâêîôûç]{4,}\b', t))
    filtered = [w for w in words if w not in stopwords]
    if not filtered:
        return 'Geral'
    top = Counter(filtered).most_common(3)
    return ' / '.join(w.title() for w, _ in top)


def group_requirements(requirements: list[dict], n_groups: int = None) -> list[dict]:
    """Agrupa requisitos por tema usando K-means sobre embeddings semânticos.

    n_groups: número de grupos (auto se None — heurística: max(2, n//5), cap 7).
    Retorna a lista original com campo 'group_id' e 'group_label' adicionados.
    """
    n = len(requirements)
    if n < 3:
        for r in requirements:
            r['group_id']    = 0
            r['group_label'] = 'Geral'
        return requirements

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np

        if n_groups is None:
            n_groups = min(7, max(2, n // 5))

        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [r['text'] for r in requirements]
        embs  = model.encode(texts, show_progress_bar=False)

        km = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        labels = km.fit_predict(embs)

        # Gera rótulos automáticos por cluster
        cluster_texts = {i: [] for i in range(n_groups)}
        for idx, label in enumerate(labels):
            cluster_texts[label].append(texts[idx])

        cluster_labels = {
            i: _auto_label_group(cluster_texts[i])
            for i in range(n_groups)
        }

        for req, label in zip(requirements, labels):
            req['group_id']    = int(label)
            req['group_label'] = cluster_labels[int(label)]

        return requirements

    except Exception as e:
        print(f"[requirements_analyzer] group_requirements falhou: {e}")
        for r in requirements:
            r['group_id']    = 0
            r['group_label'] = 'Geral'
        return requirements
