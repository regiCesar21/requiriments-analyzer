from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from sentence_transformers import SentenceTransformer
from helpers.classification_score_intent import map_score_to_label

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

# Inicialize os pipelines (uma vez)
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


def create_directory_matrix(type_train, type_directory, base_name):
    root_image_path = Path(join(base_path, type_directory,type_train))
    # List only folders
    folders = [p for p in root_image_path.iterdir() if p.is_dir()]
    # Regular expression to extract numbers at the end of the folder name
    pattern = re.compile(f'{base_name}(\d+)$')
    # Extract the numbers from the existing folders
    numbers = []
    for folder in folders:
        match = pattern.match(folder.name)
        if match:
            numbers.append(int(match.group(1)))

    # Determine the next number
    next_num = max(numbers, default=0) + 1
    new_folder = root_image_path / f"{base_name}{next_num}"

    # Create the new folder
    new_folder.mkdir(exist_ok=True)
    print(f"Folder created: {new_folder}")

    return new_folder



def create_matriz_confusion(predictions, y_test, name_model, type_train, output_dir):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão: {name_model}")
    # plt.show()

    image_path = join(base_path, "image",type_train,output_dir)
    # Gerar nome do arquivo (sem espaços)
    filename = f"{name_model.replace(' ', '_').lower()}_confusion_matrix.png"
    filepath = join(image_path, filename)
    # Salvar a imagem
    plt.savefig(filepath, dpi=300, bbox_inches='tight')


def build_pipeline_complete(x, y, type_train):
    
    # 2. Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    # Modelos para testar
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100), "Random Forest")
    ]

    output_dir = create_directory_matrix(type_train, type_directory="image", base_name="matrices")
    version_directory = create_directory_matrix(type_train, type_directory="model_train", base_name="version")
    # Treino e avaliação
    for model, name in models:
        predictions = train_and_evaluate(model, name, X_train, X_test, y_train, y_test,type_train,version_directory)
        create_matriz_confusion(predictions, y_test, name, type_train,output_dir)
        print(f"| ### The Model: {name} ### |")
        print(classification_report(y_test, predictions, digits=3))
        save_classification_report(name, y_test, predictions)


def build_pipeline(df):

    df['maturity_label'] = df['maturity_score'].apply(map_score_to_label)

    # Divisão dos dados
    x = df['text_clean']
    y = df['maturity_label']

    # 2. Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Modelos para testar
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100), "Random Forest")
    ]

    type_train = "model_train_maturity_score"
    output_dir = create_directory_matrix(type_train, type_directory="image", base_name="matrices")
    version_directory = create_directory_matrix(type_train, type_directory="model_train", base_name="version")
    # Treino e avaliação
    for model, name in models:
       predictions = train_and_evaluate(model, name, X_train, X_test, y_train, y_test,type_train,version_directory)
       create_matriz_confusion(predictions, y_test, name, type_train,output_dir)
       print(f"| ### The Model: {name} ### |")
       print(classification_report(y_test, predictions, digits=3))

def train_intent_classifier(df):

    intent_counts = df['intent'].value_counts()
    valid_intents = intent_counts[intent_counts >= 2].index
    df_filtered = df[df['intent'].isin(valid_intents)]

    # Divisão dos dados
    x = df_filtered['text']
    y = df_filtered['intent']

    # 2. Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    # Modelos para testar
    models = [
        (MultinomialNB(), "Naive Bayes"),
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100), "Random Forest")
    ]

    type_train = "model_train_intent"
    output_dir = create_directory_matrix(type_train, type_directory="image", base_name="matrices")
    version_directory = create_directory_matrix(type_train, type_directory="model_train", base_name="version")
    # Treino e avaliação
    for model, name in models:
        predictions = train_and_evaluate(model, name, X_train, X_test, y_train, y_test,type_train,version_directory)
        create_matriz_confusion(predictions, y_test, name, type_train,output_dir)
        print(f"| ### The Model: {name} ### |")
        print(classification_report(y_test, predictions, digits=3))
        save_classification_report(name, y_test, predictions)

# Função de avaliação modificada para incluir métricas
def train_and_evaluate(model, model_name, x_train, x_test, y_train, y_test, type_train,version_directory):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', model)
    ])

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    filename = model_name.replace(" ", "_").lower()


    path_model = join(base_path,"model_train",type_train,version_directory)
    joblib.dump(pipeline, join(path_model, f"{filename}_maturity_model.pkl"))
    return predictions

def save_classification_report(name, y_test, predictions):
    file_path=join(base_path,"log","classification_reports.txt")
    # Gera o relatório de classificação como string
    report = classification_report(y_test, predictions, digits=3)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"| ### The Model: {name} ### |\n")
        f.write(report + "\n")
        f.write("=" * 60 + "\n")

