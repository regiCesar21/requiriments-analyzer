"""
Avaliação quantitativa do pipeline de extração de requisitos.

Métricas calculadas:
  Etapa 1 — Filtro semântico:  Precision / Recall / F1 (requisito vs irrelevante)
  Etapa 2 — Classificação:     Precision / Recall / F1 por tipo (F / NF / BR)
  Etapa 3 — Combinada:         Precision / Recall / F1 considerando tipo correto

Documentos de teste:
  - documents/test_texto_corrido.txt  (bancário, 34 sentenças)
  - documents/test_ata_clinica.txt    (clínica médica, 27 sentenças)
"""

import sys, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from helpers.extract_text import extract_from_upload
from helpers.requirements_extractor import (
    extract_requirements, TYPE_LABELS, TYPE_ICONS,
    classify_sentence, classify_sentence_zeroshot,
    _zs_cache,
)

# ── Ground Truth — Bancário (test_texto_corrido.txt, 34 sentenças) ────────────
GROUND_TRUTH_BANCO = {
    0:  'irrelevant',    # Cabeçalho da ata (NLTK mergeia com parágrafo inicial)
    1:  'irrelevant',    # "O banco Nexus... precisa modernizar" — contexto organizacional
    2:  'irrelevant',    # "Marcos, representando..." — narrativa
    3:  'irrelevant',    # "Segundo Marcos, os clientes reclamam..." — problema, não req
    4:  'functional',    # "Ana anotou que o portal precisará mostrar o extrato unificado"
    5:  'non_functional',# "Carlos levantou que o sistema terá que carregar em < 3s"
    6:  'irrelevant',    # "Fernanda apresentou os resultados da pesquisa de UX."
    7:  'non_functional',# "73% dos usuários... portal tem que funcionar em móveis"
    8:  'non_functional',# "qualquer dado financeiro... criptografado — TLS 1.3"
    9:  'irrelevant',    # "A discussão avançou para o tema de transferências."
    10: 'functional',    # "os clientes precisam conseguir fazer TED e PIX"
    11: 'business_rule', # "limite R$1.000 — acima disso exigir biometria ou token"
    12: 'non_functional',# "sistema... precisará processar pelo menos 200 transações"
    13: 'business_rule', # "se errar a senha 3x... conta tem que ser bloqueada"
    14: 'business_rule', # "desbloqueio só pode ser feito pelo atendimento humano"
    15: 'non_functional',# "mensagem de bloqueio seja clara e amigável" — usabilidade
    16: 'irrelevant',    # "Houve uma discussão longa sobre notificações."
    17: 'functional',    # "portal deve enviar notificações push para movimentação > R$50"
    18: 'functional',    # "clientes querem poder configurar alertas personalizados"
    19: 'irrelevant',    # "O tema de acessibilidade foi levantado por Fernanda..."
    20: 'non_functional',# "portal precisa seguir as diretrizes WCAG 2.1 nível AA"
    21: 'non_functional',# "sistema deverá estar em conformidade com a LGPD"
    22: 'irrelevant',    # "seria legal ter um chatbot... ficou para segunda fase"
    23: 'irrelevant',    # "o MVP não inclui inteligência artificial" — decisão de escopo
    24: 'non_functional',# "sistema precisará manter logs de auditoria... 5 anos"
    25: 'irrelevant',    # "time de QA precisará de acesso a ambiente de homologação"
    26: 'irrelevant',    # "A reunião foi encerrada às 16h30."
    27: 'irrelevant',    # "Próximo encontro agendado para 17 de junho."
    28: 'irrelevant',    # Separador + cabeçalho do email
    29: 'non_functional',# "sessão de usuário precisa expirar após 15 minutos"
    30: 'irrelevant',    # "Deixar sessões abertas é um vetor de ataque..."
    31: 'non_functional',# "sistema não pode armazenar senhas em texto plano"
    32: 'irrelevant',    # "Parece óbvio, mas já vi casos..."
    33: 'non_functional',# "exportação de dados deverá ser registrada no log de auditoria"
}

# ── Ground Truth — Clínica (test_ata_clinica.txt, 27 sentenças) ───────────────
GROUND_TRUTH_CLINICA = {
    0:  'irrelevant',    # Cabeçalho + "Dr. Roberto abriu a reunião" (NLTK mergeia)
    1:  'irrelevant',    # "Cláudio explicou... 180 ligações diárias" — contexto do problema
    2:  'irrelevant',    # "Patrícia disse que o objetivo principal é lançar..." — objetivo org.
    3:  'irrelevant',    # "Amanda apresentou benchmarks... abandona com > 4 etapas"
    4:  'functional',    # "sistema deverá permitir que o paciente visualize os horários"
    5:  'non_functional',# "plataforma precisará suportar 500 agendamentos simultâneos"
    6:  'functional',    # "sistema deve enviar confirmação por email e SMS"
    7:  'non_functional',# "toda a plataforma precisará estar em conformidade com a LGPD"
    8:  'non_functional',# "prontuário não pode ser acessado por nenhum funcionário..." — seg.
    9:  'non_functional',# "sistema tem que funcionar bem em telas pequenas" — mobile
    10: 'irrelevant',    # "Houve uma discussão longa sobre cancelamentos de última hora"
    11: 'business_rule', # "se cancelar com menos de 2h → bloquear novos agendamentos 30 dias"
    12: 'non_functional',# "tempo de resposta... inferior a 1,5 segundos"
    13: 'irrelevant',    # "pacientes idosos preferem ligar" — comportamento do usuário
    14: 'non_functional',# "querem poder agendar de forma completamente intuitiva" — usab.
    15: 'non_functional',# "sistema deverá manter logs... 5 anos (CFM)"
    16: 'irrelevant',    # "Dr. Roberto perguntou sobre a possibilidade de integração"
    17: 'functional',    # "sistema deve se integrar com o prontuário via API REST"
    18: 'non_functional',# "fluxo de agendamento não pode ter mais de 3 telas" — UX
    19: 'business_rule', # "somente médicos cadastrados e aprovados... poderão ter agenda"
    20: 'irrelevant',    # "A equipe discutiu como lidar com diferentes planos de saúde"
    21: 'functional',    # "sistema deve permitir que o paciente informe o plano de saúde"
    22: 'irrelevant',    # "médicos precisam bloquear... hoje feito manualmente em planilha"
    23: 'functional',    # "sistema precisa permitir que o médico bloqueie intervalos de tempo"
    24: 'non_functional',# "mensagens de erro... precisam ser escritas de forma empática"
    25: 'irrelevant',    # "MVP contemplará apenas agendamento de consultas presenciais"
    26: 'irrelevant',    # "Dr. Roberto encerrou a reunião às 17h10"
}

REQ_TYPES = {'functional', 'non_functional', 'business_rule'}

# ── Helpers de métricas ───────────────────────────────────────────────────────

def prf(tp, fp, fn):
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def print_metrics(label, tp, fp, fn, tn=None):
    p, r, f1 = prf(tp, fp, fn)
    tn_str = f"  TN={tn}" if tn is not None else ""
    print(f"  {label:<22}  P={p:.2f}  R={r:.2f}  F1={f1:.2f}  "
          f"(TP={tp}  FP={fp}  FN={fn}{tn_str})")


# ── Helpers de avaliação ─────────────────────────────────────────────────────

def run_pipeline(sentences, use_zero_shot=False):
    """Roda o pipeline completo e retorna dict {idx: tipo_predito}."""
    extracted = extract_requirements(sentences, use_zero_shot=use_zero_shot)
    extracted_texts = {r['text']: r for r in extracted}
    results = {}
    for i, sent in enumerate(sentences):
        if sent in extracted_texts:
            results[i] = extracted_texts[sent]['type']
        else:
            results[i] = 'irrelevant'
    return results


def print_detail_table(sentences, results, ground_truth, label=""):
    ICONS  = {**TYPE_ICONS, 'irrelevant': '⬜', 'uncertain': '⚪'}
    LABELS = {**TYPE_LABELS, 'irrelevant': 'Irrelevante', 'uncertain': 'Incerto'}

    title = f"TABELA DETALHADA — {label}" if label else "TABELA DETALHADA"
    print("\n" + "=" * 100)
    print(title)
    print(f"{'#':>2}  {'GOLD':<16} {'PREDITO':<16} {'OK?':<4}  SENTENÇA")
    print("=" * 100)

    correct = 0
    errors  = []
    for i, sent in enumerate(sentences):
        gold        = ground_truth[i]
        pred        = results[i]
        gold_is_req = gold in REQ_TYPES
        pred_is_req = pred in REQ_TYPES
        ok = (not pred_is_req) if gold == 'irrelevant' else (pred == gold)
        correct += 1 if ok else 0

        note = ''
        if not ok:
            if gold == 'irrelevant' and pred_is_req:
                note = '← falso positivo'
            elif gold_is_req and not pred_is_req:
                note = '← falso negativo'
            elif gold_is_req and pred_is_req and pred != gold:
                note = f'← tipo errado (era {LABELS[gold]})'
            errors.append((i, gold, pred, sent[:60]))

        icon_g = ICONS.get(gold, '?')
        icon_p = ICONS.get(pred, '?')
        mark   = '✓' if ok else '✗'
        print(f"{i:>2}  {icon_g}{LABELS[gold]:<15} {icon_p}{LABELS.get(pred,'?'):<15} "
              f"{mark:<4}  {sent[:60]}  {note}")

    return correct, errors, LABELS


def print_metrics_block(results, ground_truth, label=""):
    title = f"MÉTRICAS — {label}" if label else "MÉTRICAS"
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Etapa 1: filtro semântico
    ftp = sum(1 for i in ground_truth if ground_truth[i] in REQ_TYPES     and results[i] in REQ_TYPES | {'uncertain'})
    ffp = sum(1 for i in ground_truth if ground_truth[i] == 'irrelevant'  and results[i] in REQ_TYPES | {'uncertain'})
    ffn = sum(1 for i in ground_truth if ground_truth[i] in REQ_TYPES     and results[i] == 'irrelevant')
    ftn = sum(1 for i in ground_truth if ground_truth[i] == 'irrelevant'  and results[i] == 'irrelevant')
    print("Etapa 1 — Filtro semântico (req vs irrelevante):")
    print_metrics("  Detecção de requisitos", ftp, ffp, ffn, ftn)

    # Etapa 2: classificação de tipo
    print("Etapa 2 — Classificação de tipo:")
    for rtype in ('functional', 'non_functional', 'business_rule'):
        tp = sum(1 for i in ground_truth if ground_truth[i] == rtype and results[i] == rtype)
        fp = sum(1 for i in ground_truth if ground_truth[i] != rtype and results[i] == rtype)
        fn = sum(1 for i in ground_truth if ground_truth[i] == rtype and results[i] != rtype)
        print_metrics(f"  {TYPE_LABELS[rtype]}", tp, fp, fn)

    # Etapa 3: combinada
    ctp = sum(1 for i in ground_truth if ground_truth[i] in REQ_TYPES    and results[i] == ground_truth[i])
    cfp = sum(1 for i in ground_truth if ground_truth[i] == 'irrelevant' and results[i] in REQ_TYPES)
    cfn = sum(1 for i in ground_truth if ground_truth[i] in REQ_TYPES    and results[i] != ground_truth[i])
    print("Etapa 3 — Pipeline completo (filtro + tipo correto):")
    print_metrics("  Pipeline completo", ctp, cfp, cfn)

    _, _, f1_filter = prf(ftp, ffp, ffn)
    _, _, f1_comb   = prf(ctp, cfp, cfn)
    return f1_filter, f1_comb, {'ftp': ftp, 'ffp': ffp, 'ffn': ffn, 'ftn': ftn,
                                 'ctp': ctp, 'cfp': cfp, 'cfn': cfn}


def evaluate_document(doc_name, ground_truth, use_zero_shot=False, label=""):
    """Carrega, processa e avalia um documento. Retorna (correct, total, errors, metrics_raw)."""
    doc = Path(__file__).parents[1] / 'documents' / doc_name
    buf = io.BytesIO(doc.read_bytes())
    buf.name = doc.name
    _, sentences = extract_from_upload(buf)

    assert len(sentences) == len(ground_truth), (
        f"[{doc_name}] Ground truth tem {len(ground_truth)} entradas mas o documento "
        f"gerou {len(sentences)} sentenças — revise os índices."
    )

    results = run_pipeline(sentences, use_zero_shot=use_zero_shot)
    correct, errors, _ = print_detail_table(sentences, results, ground_truth, label)
    f1_filt, f1_comb, raw = print_metrics_block(results, ground_truth, label)
    return correct, len(ground_truth), errors, f1_filt, f1_comb, raw


def aggregate_metrics(raw_list):
    """Soma os contadores brutos de múltiplos documentos e recalcula P/R/F1."""
    totals = {'ftp': 0, 'ffp': 0, 'ffn': 0, 'ftn': 0, 'ctp': 0, 'cfp': 0, 'cfn': 0}
    for r in raw_list:
        for k in totals:
            totals[k] += r[k]
    f1_filt = prf(totals['ftp'], totals['ffp'], totals['ffn'])[2]
    f1_comb = prf(totals['ctp'], totals['cfp'], totals['cfn'])[2]
    return f1_filt, f1_comb, totals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    docs = [
        ('test_texto_corrido.txt', GROUND_TRUTH_BANCO,   'Bancário'),
        ('test_ata_clinica.txt',   GROUND_TRUTH_CLINICA, 'Clínica'),
    ]

    # ── Modo 1: SVM ────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  MODO 1: TF-IDF + SVM (abordagem clássica)")
    print("█" * 70)

    svm_results = []
    for doc_name, gt, lbl in docs:
        correct, total, errors, f1_filt, f1_comb, raw = evaluate_document(
            doc_name, gt, use_zero_shot=False, label=f"SVM — {lbl}"
        )
        svm_results.append((lbl, correct, total, errors, f1_filt, f1_comb, raw))

    f1_filt_svm_agg, f1_comb_svm_agg, _ = aggregate_metrics([r[6] for r in svm_results])

    # ── Modo 2: Zero-Shot ──────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  MODO 2: Zero-Shot Classification — DeBERTa NLI (Hugging Face)")
    print("█" * 70)

    zs_results = []
    for doc_name, gt, lbl in docs:
        correct, total, errors, f1_filt, f1_comb, raw = evaluate_document(
            doc_name, gt, use_zero_shot=True, label=f"Zero-Shot — {lbl}"
        )
        zs_results.append((lbl, correct, total, errors, f1_filt, f1_comb, raw))

    f1_filt_zs_agg, f1_comb_zs_agg, _ = aggregate_metrics([r[6] for r in zs_results])

    # ── Comparativo final ──────────────────────────────────────────────────────
    total_all = sum(r[2] for r in svm_results)
    svm_correct_all = sum(r[1] for r in svm_results)
    zs_correct_all  = sum(r[1] for r in zs_results)

    print("\n" + "=" * 80)
    print("COMPARATIVO FINAL")
    print("=" * 80)
    print(f"  {'Abordagem':<32} {'Acurácia':<16} {'F1 Filtro':<12} {'F1 Completo'}")
    print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*10}")

    for svm_r, zs_r in zip(svm_results, zs_results):
        lbl, sc, tot, _, ff_s, fc_s, _ = svm_r
        _,   zc, _,   _, ff_z, fc_z, _ = zs_r
        print(f"\n  Documento: {lbl}  ({tot} sentenças)")
        print(f"  {'  TF-IDF + SVM':<32} {sc}/{tot} ({sc/tot*100:.1f}%)      "
              f"  {ff_s:.2f}       {fc_s:.2f}")
        print(f"  {'  Zero-Shot (DeBERTa NLI)':<32} {zc}/{tot} ({zc/tot*100:.1f}%)      "
              f"  {ff_z:.2f}       {fc_z:.2f}")

    print(f"\n  {'AGREGADO (ambos os docs)':<32} ", end="")
    print(f"{'SVM: '+str(svm_correct_all)+'/'+str(total_all)+' ('+f'{svm_correct_all/total_all*100:.1f}'+')':>16}  "
          f"  {f1_filt_svm_agg:.2f}       {f1_comb_svm_agg:.2f}")
    print(f"  {'':32} {'ZS:  '+str(zs_correct_all)+'/'+str(total_all)+' ('+f'{zs_correct_all/total_all*100:.1f}'+')':>16}  "
          f"  {f1_filt_zs_agg:.2f}       {f1_comb_zs_agg:.2f}")

    # ── Erros exclusivos de cada modo ──────────────────────────────────────────
    for svm_r, zs_r in zip(svm_results, zs_results):
        lbl = svm_r[0]
        idx_err_svm = {e[0] for e in svm_r[3]}
        idx_err_zs  = {e[0] for e in zs_r[3]}
        corrigidos  = idx_err_svm - idx_err_zs
        novos_erros = idx_err_zs  - idx_err_svm
        if corrigidos or novos_erros:
            print(f"\n  [{lbl}]")
        if corrigidos:
            print(f"    Zero-Shot corrigiu {len(corrigidos)} erro(s) do SVM: {sorted(corrigidos)}")
        if novos_erros:
            print(f"    Zero-Shot introduziu {len(novos_erros)} erro(s) novo(s): {sorted(novos_erros)}")


if __name__ == '__main__':
    main()
