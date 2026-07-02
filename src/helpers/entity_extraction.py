from spacy.pipeline import EntityRuler
import spacy

def add_custom_entities(nlp):
    patterns = [
        # 🔧 TECNOLOGIA
        {"label": "TECHNOLOGY", "pattern": "cloud"},
        {"label": "TECHNOLOGY", "pattern": "cloud computing"},
        {"label": "TECHNOLOGY", "pattern": "cloud-native"},
        {"label": "TECHNOLOGY", "pattern": "artificial intelligence"},
        {"label": "TECHNOLOGY", "pattern": "AI"},
        {"label": "TECHNOLOGY", "pattern": "gen AI"},
        {"label": "TECHNOLOGY", "pattern": "machine learning"},
        {"label": "TECHNOLOGY", "pattern": "RPA"},
        {"label": "TECHNOLOGY", "pattern": "intelligent automation"},
        {"label": "TECHNOLOGY", "pattern": "platforms"},
        {"label": "TECHNOLOGY", "pattern": "LLM"},  # Large Language Models

        # 🔄 PROCESSO
        {"label": "PROCESS", "pattern": "agile methodologies"},
        {"label": "PROCESS", "pattern": "automation"},
        {"label": "PROCESS", "pattern": "process optimization"},
        {"label": "PROCESS", "pattern": "KYC"},
        {"label": "PROCESS", "pattern": "onboarding"},
        {"label": "PROCESS", "pattern": "risk management"},
        {"label": "PROCESS", "pattern": "compliance automation"},

        # 🧠 ESTRATÉGIA
        {"label": "STRATEGY", "pattern": "roadmap"},
        {"label": "STRATEGY", "pattern": "digital transformation strategy"},
        {"label": "STRATEGY", "pattern": "open finance strategy"},
        {"label": "STRATEGY", "pattern": "sustainability strategy"},
        {"label": "STRATEGY", "pattern": "regulatory strategy"},

        # 👥 CULTURA
        {"label": "CULTURE", "pattern": "resistance to change"},
        {"label": "CULTURE", "pattern": "innovation culture"},
        {"label": "CULTURE", "pattern": "customer-centric culture"},
        {"label": "CULTURE", "pattern": "digital mindset"},

        # 📊 DADOS
        {"label": "DATA", "pattern": "data governance"},
        {"label": "DATA", "pattern": "data-driven"},
        {"label": "DATA", "pattern": "data monetization"},
        {"label": "DATA", "pattern": "customer data"},
        {"label": "DATA", "pattern": "real-time access"},

        # 🧭 SUSTENTABILIDADE
        {"label": "SUSTAINABILITY", "pattern": "ESG"},
        {"label": "SUSTAINABILITY", "pattern": "green finance"},
        {"label": "SUSTAINABILITY", "pattern": "net zero"},
        {"label": "SUSTAINABILITY", "pattern": "climate risk"},

        # 💼 NEGÓCIOS E MODELOS
        {"label": "BUSINESS_MODEL", "pattern": "embedded finance"},
        {"label": "BUSINESS_MODEL", "pattern": "open finance"},
        {"label": "BUSINESS_MODEL", "pattern": "financial inclusion"},
        {"label": "BUSINESS_MODEL", "pattern": "digital identity"},
        {"label": "BUSINESS_MODEL", "pattern": "identity wallet"}
    ]

    ruler = nlp.add_pipe("entity_ruler", last=True)
    ruler.add_patterns(patterns)
    # nlp.add_pipe(ruler, before="ner")  # insere o EntityRuler antes do NER padrão
    ruler = nlp.add_pipe("entity_ruler", last=True)

    return ruler

import spacy

def load_nlp_with_patterns(nlp):
    # nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
        {"label": "TECHNOLOGY", "pattern": "cloud"},
        {"label": "TECHNOLOGY", "pattern": "cloud computing"},
        {"label": "TECHNOLOGY", "pattern": "cloud-native"},
        {"label": "TECHNOLOGY", "pattern": "artificial intelligence"},
        {"label": "TECHNOLOGY", "pattern": "AI"},
        {"label": "TECHNOLOGY", "pattern": "gen AI"},
        {"label": "TECHNOLOGY", "pattern": "machine learning"},
        {"label": "TECHNOLOGY", "pattern": "RPA"},
        {"label": "TECHNOLOGY", "pattern": "intelligent automation"},
        {"label": "TECHNOLOGY", "pattern": "platforms"},
        {"label": "TECHNOLOGY", "pattern": "LLM"},
        {"label": "PROCESS", "pattern": "agile methodologies"},
        {"label": "PROCESS", "pattern": "automation"},
        {"label": "PROCESS", "pattern": "process optimization"},
        {"label": "PROCESS", "pattern": "KYC"},
        {"label": "PROCESS", "pattern": "onboarding"},
        {"label": "PROCESS", "pattern": "risk management"},
        {"label": "PROCESS", "pattern": "compliance automation"},
        {"label": "STRATEGY", "pattern": "roadmap"},
        {"label": "STRATEGY", "pattern": "digital transformation strategy"},
        {"label": "STRATEGY", "pattern": "open finance strategy"},
        {"label": "STRATEGY", "pattern": "sustainability strategy"},
        {"label": "STRATEGY", "pattern": "regulatory strategy"},
        {"label": "CULTURE", "pattern": "resistance to change"},
        {"label": "CULTURE", "pattern": "innovation culture"},
        {"label": "CULTURE", "pattern": "customer-centric culture"},
        {"label": "CULTURE", "pattern": "digital mindset"},
        {"label": "DATA", "pattern": "data governance"},
        {"label": "DATA", "pattern": "data-driven"},
        {"label": "DATA", "pattern": "data monetization"},
        {"label": "DATA", "pattern": "customer data"},
        {"label": "DATA", "pattern": "real-time access"},
        {"label": "SUSTAINABILITY", "pattern": "ESG"},
        {"label": "SUSTAINABILITY", "pattern": "green finance"},
        {"label": "SUSTAINABILITY", "pattern": "net zero"},
        {"label": "SUSTAINABILITY", "pattern": "climate risk"},
        {"label": "BUSINESS_MODEL", "pattern": "embedded finance"},
        {"label": "BUSINESS_MODEL", "pattern": "open finance"},
        {"label": "BUSINESS_MODEL", "pattern": "financial inclusion"},
        {"label": "BUSINESS_MODEL", "pattern": "digital identity"},
        {"label": "BUSINESS_MODEL", "pattern": "identity wallet"}
    ]

    ruler.add_patterns(patterns)
    return nlp


def extract_entities1(text, nlp, ruler):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Filtre todas as categorias relevantes (incluindo "CULTURA" e "DADOS")
        if ent.label_ in ["TECNOLOGIA", "PROCESSO", "ESTRATÉGIA", "CULTURA", "DADOS"]:
            entities.append((ent.text, ent.label_))
    return entities

def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]