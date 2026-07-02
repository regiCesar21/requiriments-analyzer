from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def find_similar_question(new_question, previous_questions):
    if not previous_questions:  # Lista vazia
        return -1, 0.0
    
    new_embedding = model.encode(new_question)
    previous_embeddings = model.encode(previous_questions)
    cosine_similarities = cosine_similarity([new_embedding], previous_embeddings)
    max_similarity = max(cosine_similarities[0])
    most_similar_index = cosine_similarities[0].argmax()
    return most_similar_index, max_similarity