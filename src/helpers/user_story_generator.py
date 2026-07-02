import re
import torch
from utils.detect_language import translate_text


def _build_prompt(req_en: str) -> str:
    return (
        "You are a business analyst. Write an Agile User Story for the requirement below.\n"
        "Use this format:\n"
        "As a [role], I want [action], so that [benefit].\n"
        "Acceptance Criteria:\n"
        "1) [criterion]\n"
        "2) [criterion]\n"
        "3) [criterion]\n\n"
        f"Requirement: {req_en}\n"
        "User Story:"
    )


def _postprocess(raw: str, req_en: str) -> str:
    # Remove prompt leakage
    for noise in ["You are a business analyst", "Use this format", "Acceptance Criteria",
                  "Requirement:", "User Story:", "Write an Agile"]:
        raw = raw.replace(noise, "")
    raw = re.sub(r'\s+', ' ', raw).strip()

    # Tenta começar a partir de "As a"
    match = re.search(r'[Aa]s a\b', raw)
    if match:
        raw = raw[match.start():]

    # Fallback mínimo se output for ruim
    if len(raw.split()) < 8:
        raw = (
            f"As a user, I want {req_en[:80]}, so that I can achieve the expected result. "
            "Acceptance Criteria: 1) The feature works as described. "
            "2) Edge cases are handled. 3) The user receives feedback."
        )
    return raw


def generate_user_story(requirement_text: str) -> str:
    """Gera uma User Story a partir de um requisito (PT ou EN).

    Traduz para EN antes de gerar com flan-t5-base e traduz o resultado de volta para PT.
    Os modelos são importados sob demanda para evitar carregamento duplicado.
    """
    from helpers.chatbot_interection import tokenizer, model_for_seqlm

    req_en = translate_text(requirement_text, 'en')
    prompt = _build_prompt(req_en)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
    with torch.no_grad():
        outputs = model_for_seqlm.generate(
            **inputs,
            max_length=400,
            min_length=50,
            num_beams=4,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
        )
    story_en = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    story_en = _postprocess(story_en, req_en)
    return translate_text(story_en, 'pt')
