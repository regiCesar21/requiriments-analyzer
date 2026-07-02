from helpers import process_LDA, process_chat
from helpers.chatbot_interection import chatbot_loop
from utils.extract_data import extract_data
from pathlib import Path
from os.path import join
import re
import pandas as pd

base_path = Path(__file__).resolve().parents[1]
def main():
    articles = ["artigo_38"]
    # articles = ["artigo_01", "artigo_02", "artigo_03", "artigo_04", "artigo_05", "artigo_06", "artigo_07", "artigo_08","artigo_09", "artigo_10"]
    # articles = ["artigo_01", "artigo_02", "artigo_03", "artigo_04", "artigo_05", "artigo_06", "artigo_07", "artigo_08","artigo_09", "artigo_10",
    #             "artigo_11", "artigo_12", "artigo_13", "artigo_14", "artigo_15", "artigo_16", "artigo_17", "artigo_18","artigo_19", "artigo_20",
    #             "artigo_21", "artigo_22", "artigo_23", "artigo_24", "artigo_25", "artigo_26", "artigo_27", "artigo_28","artigo_29", "artigo_30"]
    # Faltaou o artigo 36
    # books = ['livro_05','livro_06','livro_07','livro_08']
    while True:
        option = input(
            "Digite sua opção: "
            "[1] EXTRAIR DADOS | "
            "[2] TREINAR MODELO TD | "
            "[3] USAR MODELO TD | "
            "[4] TREINAR CLASSIFICADOR DE REQUISITOS (PURE) | "
            "[0] SAIR: "
        )
        option = int(option)
        if option == 1:
            for article in articles:
                print(f"| ### 📄 Starting extraction the article {article} ### |")
                extract_data(article)
                print(f"| ### ✅ Finish the extraction the article {article} ### |")
        elif option == 2:
            print("| ### ✅ Starting process ### |")
            process_chat.process()
        elif option == 3:
            output_path = join(base_path, 'output')
            df = pd.read_csv(join(output_path, "digital_transformation_maturity2.csv"))
            df['text_clean'] = df['text']
            chatbot_loop(df)
        elif option == 4:
            print("| ### 🏋️ Iniciando download do PURE dataset e treino do classificador de requisitos ### |")
            from helpers.train_requirements_classifier import train_requirements_classifier
            best = train_requirements_classifier()
            print(f"| ### ✅ Treino concluído — melhor modelo: {best['name']} (accuracy: {best['score']:.3f}) ### |")
            # Força recarga do modelo no extrator
            from helpers.requirements_extractor import reload_model
            reload_model()
        elif option == 0:
            print("| ### Project Finalizing ### |")
            break
        else:
            print("Opção invalida...")


if __name__=="__main__":
    main()