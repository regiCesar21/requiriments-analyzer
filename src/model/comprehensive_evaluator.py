import json
from model.chatbot_evaluator import ChatbotEvaluator
from model.human_evaluator import HumanEvaluator
from helpers.chatbot_interection import conversation_chatbot, load_resources
from dao.connection_bd import load_bd

class ComprehensiveEvaluator:
    def __init__(self, test_data):
        self.test_data = test_data
        self.auto_evaluator = ChatbotEvaluator()  # Classe anterior com métricas automáticas
        self.human_evaluator = HumanEvaluator()
        self.df = load_bd()
        self.resources = load_resources(self.df)
        self.results = []
    
    def run_automatic_evaluation(self):
        """Executa avaliação automática para todo o conjunto de teste"""
        print("INICIANDO AVALIAÇÃO AUTOMÁTICA")
        for item in self.test_data:
            generated_response = conversation_chatbot(item['question'], self.df, self.resources)
            item['generated_response'] = generated_response  # Armazena para uso posterior
            
            # Avaliar com métricas automáticas
            auto_metrics = self.auto_evaluator.evaluate(
                item['expected_response'],
                generated_response
            )
            
            self.results.append({
                "question": item['question'],
                "expected_response": item['expected_response'],
                "generated_response": generated_response,
                "auto_metrics": auto_metrics,
                "human_evaluation": None  # Preencher posteriormente
            })
        
        print("AVALIAÇÃO AUTOMÁTICA CONCLUÍDA")
        return self.results
    
    def run_human_evaluation(self, sample_size=5):
        """Seleciona amostras para avaliação humana"""
        if not self.results:
            print("Executando avaliação automática primeiro...")
            self.run_automatic_evaluation()
        
        # Selecionar amostras com as piores métricas automáticas
        sorted_results = sorted(
            self.results, 
            key=lambda x: x['auto_metrics']['cosine_similarity']
        )
        sample = sorted_results[:sample_size]
        
        print("\n" + "="*80)
        print("SELECIONANDO AMOSTRAS PARA AVALIAÇÃO HUMANA")
        print(f"Critério: Respostas com menor similaridade semântica\n")
        
        human_evaluations = []
        for item in sample:
            print(f"\nQUESTION: {item['question']}")
            human_eval = self.human_evaluator.evaluate_response(
                question=item['question'],
                reference=item['expected_response'],
                generated_response=item['generated_response']
            )
            
            item['human_evaluation'] = human_eval
            human_evaluations.append(item)
            
            # Salvar avaliação individual
            self.human_evaluator.save_evaluation({
                "question": item['question'],
                "expected_response": item['expected_response'],
                "generated_response": item['generated_response'],
                "auto_metrics": item['auto_metrics'],
                "human_evaluation": human_eval
            })
        
        # Gerar relatório consolidado
        consolidated = self.generate_combined_report()
        return consolidated

    
    def generate_combined_report(self):
        """Gera relatório combinando métricas automáticas e humanas"""
        report = {
            "summary": {
                "total_responses": len(self.results),
                "auto_metrics_avg": {},
                "human_metrics_avg": {}
            },
            "detailed_results": self.results
        }
        
        # Calcular médias automáticas
        auto_metrics = ["cosine_similarity", "rouge_l", "meteor_score", "bleu_score"]
        for metric in auto_metrics:
            values = [r['auto_metrics'][metric] for r in self.results if r['auto_metrics'][metric] >= 0]
            if values:
                report["summary"]["auto_metrics_avg"][metric] = sum(values) / len(values)
        
        # Calcular médias humanas (apenas para amostras avaliadas)
        human_metrics = ["relevance", "correctness", "completeness", "clarity"]
        human_scores = {metric: [] for metric in human_metrics}
        
        for item in self.results:
            if item['human_evaluation']:
                for metric in human_metrics:
                    score = item['human_evaluation']['scores'].get(metric)
                    if score:
                        human_scores[metric].append(score)
        
        for metric, scores in human_scores.items():
            if scores:
                report["summary"]["human_metrics_avg"][metric] = sum(scores) / len(scores)
        
        # Identificar pontos fracos
        report["weak_points"] = self.identify_weak_points()
        
        return report
    
    def identify_weak_points(self):
        """Identifica áreas problemáticas com base nas avaliações"""
        weak_points = {
            "low_semantic_similarity": [],
            "human_low_scores": []
        }
        
        # Identificar respostas com baixa similaridade semântica
        for item in self.results:
            if item['auto_metrics']['cosine_similarity'] < 0.6:
                weak_points["low_semantic_similarity"].append({
                    "question": item['question'],
                    "score": item['auto_metrics']['cosine_similarity']
                })
        
        # Identificar respostas com baixa avaliação humana
        for item in self.results:
            if item['human_evaluation']:
                low_scores = [
                    metric for metric, score in item['human_evaluation']['scores'].items() 
                    if score < 3
                ]
                if low_scores:
                    weak_points["human_low_scores"].append({
                        "question": item['question'],
                        "criteria": low_scores,
                        "comments": item['human_evaluation']['comments']
                    })
        
        return weak_points
    
    def save_full_report(self, filename="full_evaluation_report.json"):
        """Salva o relatório completo em arquivo JSON"""
        report = self.generate_combined_report()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Relatório completo salvo em: {filename}")
        return filename