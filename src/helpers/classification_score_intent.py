from helpers.preprocess import load_spacy_model, preprocess_with_spacy
from helpers.entity_extraction import extract_entities, add_custom_entities, load_nlp_with_patterns
from resource.atribute_score import groups
from resource.key_world import keywords
from os.path import join
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

# Create a custom Entity Ruler
def load_nlp():
    spacy_model = load_spacy_model()
    nlp = load_nlp_with_patterns(spacy_model)
    return nlp

def assign_entities(phrase):
    nlp = load_nlp()
    return extract_entities(phrase, nlp)

def assign_score(phrase, score_map):
    if not phrase or not phrase.strip():
        return 1
    phrase_lower = phrase.lower()
    scores_founds = []
    for term, score in score_map.items():
        if term in phrase_lower:
            scores_founds.append(score)
    return max(scores_founds) if scores_founds else 1  #1 = irrelevant by default

# Regex with word boundaries to avoid false positives like "ai" inside "said"
def assign_intent(phrase, intent_map):
    if not phrase or not phrase.strip():
        return 'others'
    phrase_lower = phrase.lower()
    intents_founds = []
    for term, intent in intent_map.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, phrase_lower):
            intents_founds.append(intent)
    return max(set(intents_founds), key=intents_founds.count) if intents_founds else "others"

def assign_intent_from_keyword(keyword, intent_map):
    word_lower = keyword.lower()
    for intent, terms in intent_map.items():
        if any(t in word_lower for t in terms):
            return intent
    return "others"


def assign_category(phrase, category_keywords):
    if not phrase or not phrase.strip():
        return 'Uncategorized'
    phrase_lower = phrase.lower()
    scores = {category: 0 for category in category_keywords}
    for category, words in category_keywords.items():
        for word in words:
            if word in phrase_lower:
                scores[category] += 1
    category_most_relevant = max(scores, key=scores.get)
    return category_most_relevant if scores[category_most_relevant] > 0 else "Uncategorized"

def extract_new_intent(phrases, new_key_words, model, df_scores_intents):
    df_intent = df_scores_intents[["keywords", "intent"]]
    existing_terms = [term[0] for term in df_intent]
    only_keywords = [kw[0] for kw in new_key_words]

    emb_existing = model.encode(existing_terms, convert_to_tensor=True)
    emb_news = model.encode(only_keywords, convert_to_tensor=True)

    # === 5. Filter new words that are not similar to existing ones (score < 0.6) ===
    suggested_terms = []
    for i, term in enumerate(only_keywords):
        score_max = util.cos_sim(emb_news[i], emb_existing).max().item()
        if score_max < 0.6:  # New enough word
            suggested_terms.append((term, score_max))

    # === 6. Cluster new terms to suggest intent groups ===
    new_terms = [termo for termo, _ in suggested_terms]
    emb_suggested = model.encode(new_terms)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(emb_suggested)

    # === 7. Organize terms by cluster ===
    clusters = {}
    for idx, label in enumerate(kmeans.labels_):
        clusters.setdefault(label, []).append(new_terms[idx])

    # === 8. Display suggested results ===
    print("\nNew potentially relevant keywords (not similar to existing ones):")
    for term, score in suggested_terms:
        print(f"- {term} (maximum similarity: {score:.2f})")

    print("\nGrouping suggestion for possible new intent:")
    for label, terms in clusters.items():
        print(f"\nSuggested Intent {label}:")
        for term in terms:
            print(f"  - {term}")

    # Group by cluster
    new_clusters = {}
    labels = kmeans.labels_
    for term, label in zip(only_keywords, labels):
        new_clusters.setdefault(label, []).append(term)

    # Generate cluster names
    clusters_names = {label: generate_cluster_name(terms, model) for label, terms in new_clusters.items()}
    term_reference = df_intent.iloc[0,0]

    keywords_filtered = []
    scores = []
    intents = []
    for (term, label) in suggested_terms:
        if pd.notna(term) and pd.notna(term_reference):
            sim = round(
                util.cos_sim(
                    model.encode([str(term)])[0],
                    model.encode([str(term_reference)])[0]
                ).item(), 2
            )
            keywords_filtered.append(term)
            scores.append(sim)
            intents.append(clusters_names[label])

    return prepare_file_new_intent(keywords_filtered,scores,intents, df_scores_intents)


def prepare_file_new_intent(keywords_filtered,intents,scores, df_scores_intents):

    # === 7. Preparer the data to salve ===
    df_final = pd.DataFrame({
        "keyword": keywords_filtered,
        "score": scores,
        "intent": intents
    })
    # === 8. Salve the CSV ===
    df_news = pd.DataFrame(df_final)
    new_df = pd.concat([df_news, df_scores_intents])
    data_path = join(base_path, 'data')
    new_df.to_csv(join(data_path, "keywords_with_scores_and_intents.csv"))

    return new_df


# Generate semantic description by cluster (automatic via most central words)
def generate_cluster_name(terms, model):
    emb_termos = model.encode(terms)
    centroid = emb_termos.mean(axis=0)

    # Calculate centroid similarity with each term
    sims = [(term, util.cos_sim(model.encode([term])[0], centroid).item()) for term in terms]
    most_representative_term = sorted(sims, key=lambda x: -x[1])[0][0]

    # Alternative: join the 2 most representative ones
    return most_representative_term.title()


def map_score_to_label(score):
    if score <= 0.33:
        return 'low'
    elif score <= 0.66:
        return 'average'
    else:
        return 'high'

