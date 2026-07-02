import json
import os
import random
from datetime import datetime

class HumanEvaluator:
    def __init__(self, save_path="human_evaluations"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def evaluate_response(self, question, reference, generated_response):
        """Realiza avaliação humana de uma resposta"""
        print("\n" + "="*80)
        print(f"PERGUNTA: {question}")
        print("\nRESPOSTA GERADA:")
        print(generated_response)
        print("\nRESPOSTA DE REFERÊNCIA:")
        print(reference)
        print("\n" + "-"*80)
        
        criteria = {
            "relevance": "A resposta atende à pergunta? (1 = Não atende, 5 = Atende perfeitamente)",
            "correctness": "As informações são corretas? (1 = Com erros, 5 = Totalmente correta)",
            "completeness": "Todos os aspectos foram abordados? (1 = Incompleta, 5 = Completa)",
            "clarity": "A resposta é fácil de entender? (1 = Confusa, 5 = Muito clara)"
        }
        
        scores = {}
        comments = {}
        
        print("AVALIE A RESPOSTA GERADA (1-5):")
        for criterion, prompt in criteria.items():
            while True:
                try:
                    score = int(input(f"{prompt}: "))
                    if 1 <= score <= 5:
                        scores[criterion] = score
                        break
                    else:
                        print("Por favor, insira um valor entre 1 e 5.")
                except ValueError:
                    print("Entrada inválida. Por favor, insira um número.")
            
            comment = input(f"Comentários sobre {criterion.replace('_', ' ')}: ")
            comments[criterion] = comment
        
        overall_comment = input("Comentários gerais sobre a resposta: ")
        
        return {
            "scores": scores,
            "comments": comments,
            "overall_comment": overall_comment,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_evaluation(self, evaluation_data, question_id=None):
        """Salva a avaliação em arquivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{question_id}_{timestamp}.json" if question_id else f"eval_{timestamp}.json"
        filepath = os.path.join(self.save_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAvaliação salva em: {filepath}")
        return filepath
    
    def batch_evaluate(self, test_data, sample_size=5):
        """Realiza avaliação humana para um conjunto de amostras"""
        evaluations = {}
        print(f"INICIANDO AVALIAÇÃO HUMANA PARA {sample_size} AMOSTRAS")
        
        sampled_items = random.sample(test_data, min(sample_size, len(test_data)))
        
        for idx, item in enumerate(sampled_items, 1):
            print(f"\n\n[AMOSTRA {idx}/{len(sampled_items)}]")
            eval_data = self.evaluate_response(
                question=item['question'],
                reference=item['expected_response'],
                generated_response=item.get('generated_response', '')  # Assume que a resposta já foi gerada
            )
            
            # Adicionar contexto
            eval_data['question'] = item['question']
            eval_data['expected_response'] = item['expected_response']
            eval_data['generated_response'] = item.get('generated_response', '')
            
            # Salvar avaliação individual
            self.save_evaluation(eval_data, f"q{idx}")
            evaluations[f"q{idx}"] = eval_data
        
        # Gerar relatório consolidado
        report = self.generate_summary_report(evaluations)
        report_path = os.path.join(self.save_path, "consolidated_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print(f"RELATÓRIO CONSOLIDADO SALVO EM: {report_path}")
        return report
    
    def generate_summary_report(self, evaluations):
        """Gera relatório consolidado das avaliações"""
        summary = {
            "total_evaluations": len(evaluations),
            "average_scores": {},
            "question_breakdown": {}
        }
        
        # Calcular médias
        criteria = ["relevance", "correctness", "completeness", "clarity"]
        total_scores = {criterion: 0 for criterion in criteria}
        
        for qid, evaluation in evaluations.items():
            summary["question_breakdown"][qid] = {
                "scores": evaluation["scores"],
                "overall_comment": evaluation["overall_comment"]
            }
            
            for criterion in criteria:
                total_scores[criterion] += evaluation["scores"][criterion]
        
        for criterion in criteria:
            summary["average_scores"][criterion] = total_scores[criterion] / len(evaluations)
        
        return summary