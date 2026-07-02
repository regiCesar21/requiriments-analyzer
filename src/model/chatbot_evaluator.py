from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk
import numpy as np
import logging

# Download de recursos necessários (executar apenas uma vez)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Configurar logging para ignorar warnings específicos
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class ChatbotEvaluator:
    def __init__(self):
        # Carregar modelo com parâmetros otimizados
        self.similarity_model = self._load_model()
        self.rouge_evaluator = Rouge()
    
    def _load_model(self):
        """Carrega o modelo com fallback para versão alternativa se necessário"""
        try:
            return SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                device='cpu',  # Forçar CPU para maior compatibilidade
                use_auth_token=False
            )
        except Exception as e:
            print(f"Erro ao carregar modelo principal: {e}")
            print("Carregando alternativa menor...")
            return SentenceTransformer('paraphrase-albert-small-v2')
    
    def evaluate(self, reference, response):
        try:
            # 1. Similaridade Semântica (métrica principal)
            emb_ref = self.similarity_model.encode([reference])
            emb_res = self.similarity_model.encode([response])
            cos_sim = cosine_similarity(emb_ref, emb_res)[0][0]
            
            # 2. Métricas Baseadas em Sobreposição
            # BLEU (n-gram precision)
            bleu = sentence_bleu(
                [reference.split()], 
                response.split(),
                weights=(0.5, 0.3, 0.2))
            
            # ROUGE-L (longest common subsequence)
            rouge_scores = self.rouge_evaluator.get_scores(response, reference)
            rouge_l = rouge_scores[0]['rouge-l']['f']
            
            # METEOR (semantic word matching)
            meteor = meteor_score(
                [reference.split()], 
                response.split()
            )
            
            # 3. Métricas de Forma
            length_diff = len(response) - len(reference)
            length_ratio = len(response) / max(len(reference), 1)
            
            return {
                # Métricas Semânticas
                "cosine_similarity": float(cos_sim),
                
                # Métricas de Sobreposição
                "rouge_l": float(rouge_l),
                "meteor_score": float(meteor),
                "bleu_score": float(bleu),
                
                # Métricas de Forma
                "length_difference": int(length_diff),
                "length_ratio": float(length_ratio)
            }
        
        except Exception as e:
            print(f"Erro na avaliação: {str(e)}")
            return {
                "cosine_similarity": -1,
                "rouge_l": -1,
                "meteor_score": -1,
                "bleu_score": -1,
                "length_difference": 0,
                "length_ratio": 0
            }