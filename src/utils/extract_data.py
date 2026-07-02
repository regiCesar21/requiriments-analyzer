import fitz  # PyMuPDF
import pandas as pd
import re
from pathlib import Path
from os.path import join
from collections import Counter
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from resource.key_world import keywords
from resource.intent_map import intent_map
from helpers.extract_text import extract_pdf_text, extract_relevant_phrases, extract_key_words
from helpers.generated_dataset import generate_dataset
from helpers.classification_score_intent import assign_intent_from_keyword
from resource.category_keywords import category_keywords
from resource.category_keywords_pt import category_keywords_pt
from utils.detect_language import identify_language

base_path = Path(__file__).resolve().parents[2]
# 4. Rodar tudo
def extract_data(article):

    input_path = join(base_path, 'input')
    output_path = join(base_path, 'output')
    path_pdf = join(input_path, article+".pdf")

    # === 2. Initialize the KeyBERT model ===
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text, metadata = extract_pdf_text(path_pdf)
    # cleaned_text = clean_texts_parallel(text)
    cleaned_text = text
    key_words = extract_key_words(kw_model, cleaned_text)

    all_keywords = []
    language = identify_language(text)
    if language == 'pt':
        all_keywords = [kw for keywords in category_keywords_pt.values() for kw in keywords]
        [all_keywords.append(kw[0]) for kw in key_words]
    if language == 'en':
        all_keywords = [kw for keywords in category_keywords.values() for kw in keywords]
        [all_keywords.append(kw[0]) for kw in key_words]

    phrases = extract_relevant_phrases(cleaned_text, all_keywords, model)

    generate_dataset(phrases, metadata, key_words, model)






