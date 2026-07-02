import fitz  # PyMuPDF
import pandas as pd
import re
from os.path import join
import os

from helpers.classification_score_intent import assign_intent, assign_score, assign_entities, extract_new_intent, assign_category
from resource.category_keywords import category_keywords
from utils.detect_language import identify_language, translate_pt_to_en, translate_auto_to_en
from dao.connection_bd import send_text_db

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

# 3. Gerar estrutura para DataFrame
def generate_dataset(sentences, metadata, new_key_words, model):
    print(f"| ### 🛠️ Generating dataset from extracted data... ### |")
    data = []
    data_path = join(base_path, 'data')
    df_scores_intents = pd.read_csv(join(data_path, "keywords_with_scores_and_intents.csv"))

    # df_scores_intents = extract_new_intent(sentences, new_key_words, model, df_scores_intents)

    score_map = {str(p).lower(): s for p, s in zip(df_scores_intents["keywords"], df_scores_intents["score"]) if pd.notna(p)}
    intent_map = {str(p).lower(): intent for p, intent in zip(df_scores_intents["keywords"], df_scores_intents["intent"]) if pd.notna(p)}
    for sentence in sentences:
        sentence = translate_auto_to_en(sentence)
        score = assign_score(sentence, score_map)
        intent = assign_intent(sentence, intent_map)
        entities = assign_entities(sentence)
        category = assign_category(sentence, category_keywords)
        data.append({
            "text": sentence,
            "intent": intent,
            "maturity_score": score,
            "entities": entities,
            "category": category,
            "metadata": metadata
        })
    # save_dataframe(pd.DataFrame(data))
    send_text_db(pd.DataFrame(data))
    print(f"| ### 📁 Dataset generation complete... ### |")

def save_dataframe(df):

    output_path = join(base_path, 'output')
    file_path = join(output_path, "digital_transformation_maturity3.csv")
    print(df['maturity_score'].value_counts())

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"✅ File saved with {len(df)} examples.")
    else:
        df_dtm = pd.read_csv(file_path)
        df_concat = pd.concat([df_dtm, df])
        df_concat.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"✅ File saved with {len(df)} examples.")

