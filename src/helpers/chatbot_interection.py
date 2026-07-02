from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from helpers.classification_score_intent import map_score_to_label
from utils.detect_language import identify_language, translate_pt_to_en, translate_auto_to_en, translate_text
import streamlit as st

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

# output_path = join(base_path, 'output')
# df = pd.read_csv(join(output_path, "digital_transformation_maturity2.csv"))

# rephrase_pipe é carregado sob demanda em improve_question() — evita falha no import
# quando a versão do transformers não suporta text2text-generation no nível de módulo
_rephrase_pipe = None
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base",legacy=True)
model_for_seqlm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.float16)
# embed_model_transformer = SentenceTransformer("paraphrase-mpnet-base-v2", device='cuda' if torch.cuda.is_available() else 'cpu')
embed_model_transformer = SentenceTransformer("paraphrase-mpnet-base-v2")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pré-calcular os IDs proibidos uma vez (fora da função)
FORBIDDEN_WORDS = ["survey", "study", "research", "report", "data", "percent", "%",
                   "found that", "according to", "research shows", "studies show"]

# Obter IDs únicos de tokens proibidos
bad_word_ids = []
for word in FORBIDDEN_WORDS:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    bad_word_ids.extend(tokens)

UNIQUE_BAD_IDS = list(set(bad_word_ids))
BAD_WORDS_IDS = [[id] for id in UNIQUE_BAD_IDS]

@st.cache_resource
# Carrega os modelos salvos
def load_models():
    intent_model_path = join(base_path, "model_train", "model_train_intent", "version1",
                             "regressão_logística_maturity_model.pkl")
    maturity_model_path = join(base_path, "model_train", "model_train_maturity_score", "version1",
                               "regressão_logística_maturity_model.pkl")

    intent_model = joblib.load(intent_model_path)
    maturity_model = joblib.load(maturity_model_path)
    return intent_model, maturity_model

def prepare_semantic_search(df):
    corpus = df["text"].tolist()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    embeddings = embed_model.encode(corpus, convert_to_tensor=True)
    return tfidf_vectorizer, tfidf_matrix, embed_model, embeddings


# TF-IDF Search
def tfidf_search(query, tfidf_vectorizer, tfidf_matrix, df, top_k=3):
    query_vec = tfidf_vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices][["text", "maturity_score", "intent"]]
    results['similarity'] = similarity[top_indices]  # Adiciona coluna auxiliar
    return results


# Semantic Search com SentenceTransformer
# Ajustando o top_k de 3 para 1
def semantic_search(query, model, embeddings, df, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=top_k)[0]
    return df.iloc[[hit['corpus_id'] for hit in hits]][["text", "maturity_score", "intent"]]


def improve_question(question):
    global _rephrase_pipe
    if _rephrase_pipe is None:
        _rephrase_pipe = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    prompt = f"paraphrase: {question} </s>"
    results = _rephrase_pipe(prompt, max_length=64, do_sample=True, top_k=50, num_return_sequences=3)
    # Escolher o mais diferente da original
    original_vec = embed_model_transformer.encode([question], convert_to_tensor=True)
    candidates = [r['generated_text'] for r in results]
    candidate_vecs = embed_model_transformer.encode(candidates, convert_to_tensor=True)
    sims = util.cos_sim(original_vec, candidate_vecs)[0]
    best_idx = torch.argmin(sims).item()
    return candidates[best_idx]


def extract_relevant_snippets(text, query, model, window_size=300, top_n=3):
    """Extrai trechos mais relevantes usando similaridade com a consulta"""
    words = text.split()
    snippets = []
    # Divide o texto em janelas
    for i in range(0, len(words), window_size // 2):
        window = " ".join(words[i:i + window_size])
        window_embed = model.encode(window)
        query_embed = model.encode(query)
        similarity = util.pytorch_cos_sim(query_embed, window_embed).item()
        snippets.append((window, similarity))
    # Seleciona os trechos mais relevantes
    snippets.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in snippets[:top_n]]


def build_context(retrieved_texts, query, max_tokens=1500):
    """Constrói contexto com trechos mais relevantes"""
    context = []
    token_count = 0
    for text in retrieved_texts:
        snippets = extract_relevant_snippets(text, query, embed_model_transformer)
        for snippet in snippets:
            snippet_tokens = len(tokenizer.tokenize(snippet))
            if token_count + snippet_tokens <= max_tokens:
                context.append(snippet)
                token_count += snippet_tokens
            else:
                break
        if token_count >= max_tokens:
            break
    return "\n\n".join(context)


def truncate_context(texts, max_tokens=500):
    context = ""
    for t in texts:
        if not t or not isinstance(t, str):
            continue
        new_context = context + " " + t.strip()
        if len(tokenizer.tokenize(new_context)) > max_tokens:
            break
        context = new_context
    return context.strip()


def filter_context(texts):
    keywords = [
        "maturity level",
        "Practical Recommendations",
        "Strategic Importance",
        "Maturity-Based Guidance",
        "Cross-Dimensional Insight"
    ]
    return [t for t in texts if not any(kw in t for kw in keywords)]


def prompt_format(intent, maturity, context, question):
    DELIMITER = "|||||||||||||||||||||||||||||||||||||||||||||||||"
    prompt = f"""
       You are a professional assistant specialized in digital transformation.
       You will receive:
           - An **intent**: the main topic being asked about (e.g. Customer Experience, Operational Processes, Technology and Infrastructure, Culture and People, Business Models).
           - A **maturity level**: indicating the organization's current stage in digital maturity (Emerging, Developing, Maturing, Leading).
           - A **question**: the user's inquiry regarding digital transformation.
           - A **context**: optional insights from documents, research, or prior conversation relevant to the question.

       Your job is to provide a clear, well-structured, and insightful answer that helps guide the user.
        |||| START STRUCTURE ||||
       Structure your response like this:
       1. **Strategic Importance**  
          Explain why the intent/topic is critical in the broader context of digital transformation.
       2. **Maturity-Based Guidance**  
          Tailor your response to the organization’s maturity level.  
          - For *Emerging* organizations: Focus on awareness, foundation building, and early steps.  
          - For *Developing*: Emphasize integration, consistency, and internal alignment.  
          - For *Maturing*: Highlight optimization, scalability, and performance.  
          - For *Leading*: Encourage innovation, ecosystem orchestration, and competitive differentiation.
       3. **Practical Recommendations**  
          Give actionable guidance, concrete examples, or tools/methods that can be applied in this area.
       4. **Cross-Dimensional Insight**  
          If relevant, explain how this dimension connects with others (e.g., how culture enables customer experience, or how tech supports operations).
       Use a professional, consultative tone. Avoid generic advice. Make your recommendations practical and tailored to the scenario.
       |||| END STRUCTURE ||||

       Intent: {intent}
       Maturity Level: {maturity}
       Context: {context}
       Question: {question}
       Answer:
       """

    return prompt


def test_prompt(intent, maturity, context, question):
    DELIMITER = "||||||||||||||||||||||||||||||||||||||||||||||||||||||"
    prompt = f"""
    {DELIMITER}
    ## INTERNAL INSTRUCTIONS (DO NOT INCLUDE IN THE ANSWER) ##

    You are a professional assistant specialized in **digital transformation for organizations**.

    You will receive the following inputs:
    - **Intent**: the main topic the user is asking about (e.g., Customer Experience, Operational Processes, Technology and Infrastructure, Culture and People, Business Models).
    - **Maturity Level**: the organization’s current stage in digital maturity (Emerging, Developing, Maturing, Leading).
    - **Question**: the user’s inquiry regarding digital transformation.
    - **Context**: optional, relevant background such as document excerpts, past interactions, or domain-specific knowledge.

    🎯 **Your goal** is to generate a clear, insightful, and practical response that aligns with the topic, context, and maturity level of the organization.

    🧭 **If the question asks about the company's maturity level given a situation, first determine the maturity stage (Emerging, Developing, Maturing, Leading) and then explain your reasoning in 2-3 sentences.**

    Below are examples to guide your style and reasoning:

    ### EXAMPLES

    **Example 1**
    Situation: Temos processos totalmente analógicos sem nenhum sistema digital.
    Answer: The company's digital maturity level is **Emerging**. This is because it does not yet have digital processes implemented, indicating it is at the foundational stage of digital transformation.

    **Example 2**
    Situation: Usamos sistemas digitais em alguns processos, mas não de forma integrada, e nossa cultura é resistente a mudanças tecnológicas.
    Answer: The company's digital maturity level is **Developing**. It has started adopting digital processes but lacks integration and cultural readiness, placing it at an intermediate stage.

    **Example 3**
    Situation: Temos processos digitais bem implementados e uma cultura que incentiva a inovação tecnológica.
    Answer: The company's digital maturity level is **Maturing**. There is strong process digitization and cultural support, indicating advanced maturity and readiness to scale.

    **Example 4**
    Situation: Nossa empresa lidera o mercado em inovação digital, possui processos automatizados, cultura ágil e uso extensivo de dados para decisões estratégicas.
    Answer: The company's digital maturity level is **Leading**. It shows full integration of digital technologies and culture, enabling market leadership.

    ### END OF EXAMPLES

    ## END OF INSTRUCTIONS ##
    {DELIMITER}

    Intent: {intent}
    Maturity Level: {maturity}
    Context: {context}
    Question: {question}

    Answer:
    """
    return prompt


def generate_answer(intent, maturity, retrieved_texts, question):
    context = truncate_context(retrieved_texts)
    # context = filter_context(context)

    prompt = prompt_format(intent, maturity, context, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    outputs = model_for_seqlm.generate(
        **inputs,
        max_length=600,
        min_length=100,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.5,
        num_beams=3,
        do_sample=False,
        no_repeat_ngram_size=3,
        early_stopping=True
        # num_return_sequences=1
        # bad_words_ids=BAD_WORDS_IDS + [
        #     [tokenizer.encode("rule")[0]],
        #     [tokenizer.encode("instruction")[0]],
        #     [tokenizer.encode("context")[0]],
        #     [tokenizer.encode("guideline")[0]]
        # ]
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response = clean_response(response)
    return postprocess_response(response)


def clean_response(response):
    # Remove qualquer lista numerada ou com marcadores
    response = re.sub(r'\d+\.\s+[A-Za-z]+\s+".*?"', '', response)
    # Remove referências à estrutura interna
    structure_terms = [
        "intent",
        "maturity level",
        "Maturity-Based Guidance",
        "Practical Recommendations"
    ]
    for term in structure_terms:
        response = response.replace(term, "")
    # Remove frases incompletas
    return '. '.join([s for s in response.split('. ') if len(s.split()) > 3])
    #
    # Remove qualquer menção a regras ou estrutura
    # forbidden_phrases = [
    #     "rule", "instruction", "context", "guideline",
    #     "don't have enough", "knowledge", "provided"
    # ]

    # for phrase in forbidden_phrases:
    #     response = response.replace(phrase, "")

    # # Remove frases incompletas
    # sentences = [s for s in response.split('.') if len(s.split()) > 4]

    # # Filtro final de qualidade
    # if any(word in response for word in forbidden_phrases) or len(sentences) < 2:
    #     return "I can provide guidance on assessing digital transformation readiness. Key methods include maturity assessments, technology audits, and skills gap analysis."

    # return '. '.join(sentences).strip()


def postprocess_response(response):
    """Remove repetições óbvias na resposta final"""
    sentences = response.split('.')
    unique_sentences = []
    seen = set()
    for s in sentences:
        clean_s = s.strip()
        if clean_s and clean_s not in seen:
            seen.add(clean_s)
            unique_sentences.append(s)
    return '.'.join(unique_sentences).strip()


# Aumentar top_k e combinar TF-IDF + Semântica
def get_context(query, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings, df, top_k=15, min_similarity=0.4):
    # # Busca semântica com mais resultados
    # sem_results = semantic_search(query, embed_model, embeddings, df, top_k=top_k)

    # Busca TF-IDF com mais resultados
    tfidf_results = tfidf_search(query, tfidf_vectorizer, tfidf_matrix, df, top_k=top_k)

    # Adiciona colunas de similaridade manualmente
    # sem_results['semantic_similarity'] = [hit['score'] for hit in util.semantic_search(
    #     embed_model.encode(query, convert_to_tensor=True),
    #     embeddings,
    #     top_k=top_k
    # )[0]]

    tfidf_results['tfidf_similarity'] = cosine_similarity(
        tfidf_vectorizer.transform([query]),
        tfidf_matrix[tfidf_results.index]
    ).flatten()

    # Combina e remove duplicatas
    combined = pd.concat([tfidf_results]).drop_duplicates(subset=['text'])

    # Calcula score combinado
    # combined['combined_score'] = combined.apply(
    #     lambda row: (row['semantic_similarity'] * 0.7 + row['tfidf_similarity'] * 0.3),
    #     axis=1
    # )
    combined['combined_score'] = combined.apply(
        lambda row: (row['tfidf_similarity'] * 0.3),
        axis=1
    )

    # Filtra por similaridade mínima
    combined = combined[combined['combined_score'] > min_similarity]

    # Ordena e seleciona os melhores
    combined = combined.sort_values('combined_score', ascending=False).head(top_k)

    return combined['text'].tolist()

@st.cache_resource
def load_resources(df):
    intent_model, maturity_model = load_models()
    if df is None or df.empty or 'text' not in df.columns:
        # BD offline — sem corpus para TF-IDF/embeddings
        return {
            "tfidf": (None, None),
            "embeddings": (embed_model, None),
            "models": (intent_model, maturity_model),
        }
    tfidf_vectorizer, tfidf_matrix, _, embeddings = prepare_semantic_search(df)
    return {
        "tfidf": (tfidf_vectorizer, tfidf_matrix),
        "embeddings": (embed_model, embeddings),
        "models": (intent_model, maturity_model),
    }

def search_uploaded_docs(query, sentences, model, top_k=5):
    """Retorna as sentenças mais relevantes dos documentos enviados pelo usuário."""
    if not sentences:
        return []
    top_k = min(top_k, len(sentences))
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    return [sentences[i] for i in top_indices]


def conversation_chatbot(user_input, df, resources, uploaded_sentences=None):
    try:
        tfidf_vectorizer, tfidf_matrix = resources["tfidf"]
        embed_model, embeddings = resources["embeddings"]
        intent_model, maturity_model = resources["models"]

        user_input = translate_text(user_input, 'en')

        # Busca no corpus do BD apenas se disponível
        if tfidf_vectorizer is not None and embeddings is not None and not df.empty:
            df['text_clean'] = df['text']
            retrieved_texts = get_context(user_input, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings, df)
        else:
            retrieved_texts = []

        # Acrescenta trechos relevantes dos documentos enviados pelo usuário
        if uploaded_sentences:
            doc_snippets = search_uploaded_docs(user_input, uploaded_sentences, embed_model, top_k=5)
            retrieved_texts = doc_snippets + retrieved_texts

        context = build_context(retrieved_texts, user_input, max_tokens=1500)

        predicted_intent = intent_model.predict([user_input])[0]
        predicted_maturity = maturity_model.predict([user_input])[0]

        response = generate_answer(predicted_intent, predicted_maturity, context, user_input)
        response = response.replace('\n', ' ').strip()
        response = translate_text(response, 'pt')
        return response
    except Exception as e:
        st.error("Erro ao gerar resposta.")
        st.exception(e)

# Função do chatbot
def chatbot_loop(df):
    tfidf_vectorizer, tfidf_matrix, embed_model, embeddings = prepare_semantic_search(df)

    intent_model, maturity_model = load_models()
    print("\n🔹 Chatbot sobre Transformação Digital (digite 'sair' para encerrar)\n")
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Encerrando o chatbot. Até mais!")
            break
        user_input = improve_question(user_input)

        predicted_intent = intent_model.predict([user_input])[0]
        predicted_maturity = maturity_model.predict([user_input])[0]

        print(f"\n🎯 Intenção Detectada: {predicted_intent}")
        print(f"📈 Nível de Maturidade: {predicted_maturity}")

        print("\n🔍 Resultados mais relevantes (semânticos):")
        # results = semantic_search(user_input, embed_model, embeddings, df)

        retrieved_texts = get_context(user_input, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings, df)

        context = build_context(retrieved_texts, user_input, max_tokens=1500)

        # retrieved_texts = results['text'].tolist()
        response = generate_answer(predicted_intent, predicted_maturity, context, user_input)
        response = response.replace('\n', ' ').strip()
        print(f"\n💬 Resposta: {response}")
        # for idx, row in results.iterrows():
        #     print(f"\n📝 Texto: {row['text'][:300]}...")
        #     print(f"📈 Maturity Score: {row['maturity_score']} | 🎯 Intent: {row['intent']}")

# ANOTAÇÕES
# MODELOS NER
# Whispers tiny (Para trancrever os textos em audios ou vise-versa)
# Idea de fluxo:
# [1] Inicio
# [2] Captura da Pergunta
# [3] Pré-processamento da linguagem
# [4] Classificação e Analyze
# [5] Busca da resposta
# [6] Resposta ao usuário
# [7] feedback
# [8] Aprendizado continuo
