from deep_translator import GoogleTranslator
from langid import langid


def translate_auto_to_en(text):
    if not text or len(text.strip()) < 2:
        return text
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"[⚠️ Tradução falhou]: '{text}' — {e}")
        return text

# def translate_text(text, target_lang='en'):
#     if not text or len(text.strip()) < 2:
#         return text
#     try:
#         return GoogleTranslator(source='auto', target=target_lang).translate(text)
#     except Exception as e:
#         print(f"[⚠️ Tradução falhou]: '{text}' — {e}")
#         return text

def translate_text(text, target_lang='en', chunk_size=4000):
    if not text or len(text.strip()) < 2:
        return text

    translated_text = ""
    try:
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            translated_chunk = GoogleTranslator(source='auto', target=target_lang).translate(chunk)
            translated_text += translated_chunk
        return translated_text
    except Exception as e:
        print(f"[⚠️ Tradução falhou]: '{text}' — {e}")
        return text

def translate_pt_to_en(text):
    return GoogleTranslator(source='pt', target='en').translate(text)

def identify_language(text):
    lang, prob = langid.classify(text)
    return lang