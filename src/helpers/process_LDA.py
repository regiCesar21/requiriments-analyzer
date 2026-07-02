import nltk
import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import spacy

documents = [
    "O presidente anunciou novas medidas econômicas para conter a inflação presidente.",
    "A seleção brasileira venceu a Argentina em um jogo emocionante.",
    "Novas tecnologias estão transformando a indústria automotiva.",
    "O aumento dos preços dos combustíveis preocupa os consumidores.",
    "A pandemia de COVID-19 trouxe desafios sem precedentes para a saúde pública.",
    "A empresa lançou um novo smartphone com tecnologia de ponta.",
    "O congresso aprovou uma reforma tributária para simplificar o sistema de impostos.",
    "Pesquisadores descobriram uma nova espécie de dinossauro na América do Sul.",
    "O mercado de ações caiu após a divulgação de dados econômicos negativos.",
    "A crise climática exige ações urgentes de todos os países.",
    "A inteligência artificial está revolucionando o setor financeiro.",
    "A vacina contra a gripe estará disponível nas clínicas a partir de outubro.",
    "Os cientistas estão desenvolvendo novos tratamentos para o câncer.",
    "O turismo espacial se torna uma realidade com novos voos comerciais.",
    "O governo anunciou um plano para melhorar a infraestrutura das estradas.",
    "O festival de cinema internacional atraiu milhares de espectadores.",
    "A educação à distância ganha popularidade entre estudantes e professores.",
    "O desemprego atinge níveis recordes em várias regiões do país.",
    "Os esportes eletrônicos se tornam uma grande indústria de entretenimento.",
    "O uso de energias renováveis cresce em todo o mundo.",
    "A biodiversidade está ameaçada pela destruição de habitats naturais.",
    "O desenvolvimento sustentável é crucial para o futuro do planeta.",
    "As startups de tecnologia recebem investimentos milionários.",
    "Os serviços de streaming mudaram a forma como consumimos mídia.",
    "A robótica avança com a criação de novos robôs autônomos.",
    "Os direitos humanos são fundamentais para uma sociedade justa.",
    "A exploração espacial continua com novas missões a Marte.",
    "A música clássica ainda encanta muitas pessoas ao redor do mundo.",
    "A literatura brasileira tem ganhado destaque internacional.",
    "O comércio eletrônico cresce rapidamente com a pandemia.",
    "A reciclagem é essencial para reduzir o impacto ambiental.",
    "A saúde mental é uma prioridade na sociedade moderna.",
    "O transporte público precisa de melhorias para atender à demanda.",
    "Os avanços na medicina prolongam a expectativa de vida.",
    "A cibersegurança se torna uma preocupação crescente para empresas.",
    "A arquitetura moderna incorpora tecnologias sustentáveis.",
    "Os jogos de tabuleiro voltam a ganhar popularidade entre jovens.",
    "A agricultura precisa de inovação para alimentar a população crescente.",
    "Os oceanos enfrentam problemas graves de poluição.",
    "A moda sustentável se torna uma tendência global.",
    "A energia solar é uma solução promissora para a crise energética.",
    "O voluntariado ajuda a fortalecer as comunidades locais.",
    "A democracia é essencial para a liberdade e igualdade.",
    "A nanotecnologia oferece novas possibilidades para a indústria.",
    "O cinema independente produz filmes inovadores e criativos.",
    "A igualdade de gênero é fundamental para o progresso social.",
    "Os veículos elétricos são o futuro do transporte.",
    "A preservação da história é importante para a identidade cultural.",
    "A educação financeira é crucial para o bem-estar econômico.",
    "A arte urbana transforma paisagens e engaja a comunidade.",
    "Os direitos dos animais são uma questão de ética e justiça.",
    "A inovação científica impulsiona o desenvolvimento econômico.",
    "A mobilidade urbana precisa de soluções inteligentes.",
    "O design thinking é uma abordagem eficaz para resolver problemas complexos.",
    "A inteligência emocional é importante para o sucesso pessoal e profissional.",
    "A segurança alimentar é um desafio global.",
    "Os esportes promovem a saúde e a integração social.",
    "A cooperação internacional é vital para a paz mundial.",
    "A economia circular é uma alternativa ao modelo tradicional de produção.",
    "A programação é uma habilidade valiosa no mercado de trabalho atual.",
    "A ética profissional é essencial em todas as áreas de atuação.",
    "A leitura é fundamental para o desenvolvimento cognitivo.",
    "A cultura digital está mudando a forma como interagimos."
]

def preprocess_with_spacy(texts, nlp):
    """
    Pré-processa uma lista de textos usando o spaCy.
    A função aplica as seguintes etapas:

    1. Converte o texto para minúsculas.
    2. Tokeniza o texto usando o pipeline do spaCy.
    3. Remove stop words definidas pelo spaCy.
    4. Aplica a lematização para reduzir as palavras à sua forma base.

    Args:
        texts (list of str): Lista de documentos em texto bruto.

    Returns:
        list of list of str: Lista de documentos processados, com tokens lematizados.
    """
    processed_texts = []

    for doc in texts:
        # Processar o texto com spaCy
        spacy_doc = nlp(doc.lower())
        # Gerar lista de tokens processados
        tokens = [
            token.lemma_  # Obtém o lema do token
            for token in spacy_doc
            if not token.is_stop and not token.is_punct  # Remove stop words e pontuação
        ]
        processed_texts.append(tokens)

    return processed_texts
# Função para treinar o modelo LDA e calcular coerência e perplexidade
def train_lda_and_evaluate(dictionary, corpus, texts, num_topics, passes=15):
    # corpus: No Gensim, um corpus é normalmente uma lista de listas de tuplas, onde cada
    #  tupla representa um termo (identificado por seu ID) e sua frequência no documento.
    # id2word:Este é o dicionário que mapeia os IDs das palavras para suas formas textuais.
    # num_topics: Este parâmetro especifica o número de tópicos que o modelo LDA deve encontrar no corpus.
    # passes: Este parâmetro define o número de passes que o algoritmo deve fazer sobre o corpus durante o treinamento.
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

    # Coerência do modelo: Porque usar Coerência c_v
    #Segmentação de Palavras: Divide a lista de palavras de um tópico em pares de palavras (ou grupos maiores).
    #Medição de Co-ocorrência: Avalia a frequência com que essas palavras aparecem juntas nos documentos.
    #Similaridade de Vetores: Utiliza a similaridade de vetores para medir a proximidade semântica entre as palavras de um tópico.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    # Perplexidade do modelo
    perplexity = lda_model.log_perplexity(corpus)

    return lda_model, coherence_lda, perplexity

def compute_coherence_perplexity_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    perplexity_values = []
    for num_topics in range(start, limit, step):
        lda_model, coherence, perplexity = train_lda_and_evaluate(dictionary, corpus, texts, num_topics)
        coherence_values.append(coherence)
        perplexity_values.append(perplexity)
    return coherence_values, perplexity_values


def process():

    # Baixe os recursos do NLTK necessários
    nltk.download('stopwords')
    nltk.download('rslp')
    # Definindo palavras de parada em português
    stop_words = set(stopwords.words('portuguese'))
    nlp = spacy.load('pt_core_news_lg')
    # Aplicar o pré-processamento com spaCy
    print("✅ Aplicar o pré-processamento com spaCy")
    processed_texts = preprocess_with_spacy(documents, nlp)
    print(processed_texts)

    # Criação do dicionário e do corpus
    #Dictionary: Mapeia cada palavra única para um ID único.
    print("✅ Criação do dicionário e do corpus")
    dictionary = corpora.Dictionary(processed_texts)
    #Converte cada documento (lista de palavras) em uma lista de pares (word_id, word_count).
    #doc2bow significa "document to bag-of-words" (documento para saco de palavras).
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    # print(dictionary)
    # print(corpus)

    # Treinamento do modelo LDA com 7 tópicos (como exemplo inicial)
    print('✅ Treinamento do modelo LDA')
    lda_model, coherence_lda, perplexity = train_lda_and_evaluate(dictionary, corpus, processed_texts, num_topics=7)

    # Impressão dos tópicos gerados
    #lda_model.print_topics(): Este método do objeto lda_model retorna uma lista dos tópicos gerados pelo modelo.
    #Cada tópico é representado como uma lista de palavras com pesos associados, indicando a importância de cada palavra no tópico.
    print(lda_model.print_topics())

    # Exibindo métricas
    print('\nCoherence Score: ', coherence_lda)
    print('Perplexity: ', perplexity)

    limit = 10; start = 2; step = 1
    coherence_values, perplexity_values = compute_coherence_perplexity_values(dictionary, corpus, processed_texts, start=start, limit=limit, step=step)

    # Mostrar gráficos
    print('✅ Mostrar gráficos')
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Número de Tópicos')
    ax1.set_ylabel('Coerência', color='tab:blue')
    ax1.plot(range(start, limit, step), coherence_values, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Perplexidade', color='tab:red')
    ax2.plot(range(start, limit, step), perplexity_values, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.show()