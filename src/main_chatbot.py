import streamlit as st
import time
import uuid
import json
from io import BytesIO, StringIO
from datetime import datetime
from queue import Queue
from collections import deque
from helpers.chatbot_interection import conversation_chatbot, load_resources
from helpers.extract_text import extract_from_upload
from helpers.requirements_extractor import extract_requirements, TYPE_LABELS, TYPE_ICONS, using_ml_model
from helpers.requirements_analyzer import group_requirements, find_duplicates
from helpers.user_story_generator import generate_user_story
from helpers.audio import transcrever_audio, texto_para_audio
from helpers.audio import GravadorAudio
from dao  import connection_bd
from utils import similarity_text, interaction
import pandas as pd
import soundfile as sf


def historic():
    # Exibe todo o histórico
    for message in st.session_state.chatbot_responses:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])
        else:
            with st.chat_message("user"):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])


def display_incremental_response(text):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for chunk in text.split():
            full_text += chunk + " "
            placeholder.markdown(full_text + "▌")
            time.sleep(0.05)
        placeholder.markdown(full_text)
    return full_text


@st.cache_data(show_spinner="Carregando dados do banco...")
def load_bd():
    try:
        bd = connection_bd.find_all()
        df = pd.DataFrame(bd)
        if df.empty or 'text' not in df.columns:
            return pd.DataFrame(columns=['text', 'maturity_score', 'intent'])
        return df
    except Exception:
        return pd.DataFrame(columns=['text', 'maturity_score', 'intent'])


def main():
    st.title("TransforMind 🎈")
    
    audio_bk = None

    # Inicializações
    if "resources" not in st.session_state:
        df = load_bd()
        st.session_state.df = df
        with st.spinner("Preparando modelos..."):
            st.session_state.resources = load_resources(df)
    if "chatbot_responses" not in st.session_state:
        st.session_state.chatbot_responses = deque()
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = str(uuid.uuid4())
        st.session_state['start_time'] = time.time()
    # isQuestionAudio = False 
    # isResponseAudio = False
    if 'isQuestionAudio' not in st.session_state:
        st.session_state.isQuestionAudio = False
    if 'isResponseAudio' not in st.session_state:
        st.session_state.isResponseAudio = False
    # Inicialização do gravador
    if 'gravador' not in st.session_state:
        st.session_state.gravador = GravadorAudio()
    
    # Inicialização de session_state
    if 'ultima_transcricao' not in st.session_state:
        st.session_state.ultima_transcricao = ""
    if 'tempo_audio' not in st.session_state:
        st.session_state.tempo_audio = 0
    if 'tempo_transcricao' not in st.session_state:
        st.session_state.tempo_transcricao = 0
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = None
    if 'pergunta_isaudio' not in st.session_state:
        st.session_state.pergunta_isaudio = False
    if 'audio_gravado' not in st.session_state:
        st.session_state.pergunta_isaudio = None
    if 'ultimo_audio' not in st.session_state:
        st.session_state.ultimo_audio = None
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'extracted_requirements' not in st.session_state:
        st.session_state.extracted_requirements = []
    if 'user_stories' not in st.session_state:
        st.session_state.user_stories = {}
    if 'requirements_analyzed' not in st.session_state:
        st.session_state.requirements_analyzed = False
    if 'requirements_grouped' not in st.session_state:
        st.session_state.requirements_grouped = False
    if 'requirements_duplicates' not in st.session_state:
        st.session_state.requirements_duplicates = []
    if 'requirements_dup_run' not in st.session_state:
        st.session_state.requirements_dup_run = False
    
    with st.sidebar:
        st.title("Opções:")

        on = st.toggle("Ativar respostas em audio")

        # Botões de controle
        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🎤 Iniciar Gravação", disabled=st.session_state.gravador.gravando):
                st.session_state.tempo_inicio = st.session_state.gravador.iniciar()
                st.rerun()

            if st.button("⏹️ Parar Gravação", disabled=not st.session_state.gravador.gravando):
                st.session_state["processar_audio"] = True

        with col2:
            if st.session_state.gravador.gravando:
                tempo_decorrido = time.time() - st.session_state.get('tempo_inicio', time.time())
                st.info(f"**Gravando...** Tempo: {tempo_decorrido:.1f} segundos")
                st.progress(min(tempo_decorrido / 60, 1.0))
            else:
                st.info("Pronto para gravar")

        st.divider()
        st.subheader("Documentos")

        uploaded_file = st.file_uploader(
            "Carregar documento (PDF, TXT ou DOCX)",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            already_loaded = any(d["name"] == uploaded_file.name for d in st.session_state.uploaded_docs)
            if not already_loaded:
                with st.spinner(f"Extraindo texto de '{uploaded_file.name}'..."):
                    try:
                        _, sentences = extract_from_upload(uploaded_file)
                        st.session_state.uploaded_docs.append({
                            "name": uploaded_file.name,
                            "sentences": sentences,
                        })
                        st.success(f"'{uploaded_file.name}' carregado — {len(sentences)} sentenças indexadas.")
                    except ValueError as e:
                        st.error(str(e))

        if st.session_state.uploaded_docs:
            st.caption("Documentos carregados:")
            for i, doc in enumerate(st.session_state.uploaded_docs):
                col_doc, col_rm = st.columns([4, 1])
                with col_doc:
                    st.caption(f"- {doc['name']} ({len(doc['sentences'])} sent.)")
                with col_rm:
                    if st.button("X", key=f"rm_doc_{i}"):
                        st.session_state.uploaded_docs.pop(i)
                        st.session_state.extracted_requirements = []
                        st.session_state.user_stories = {}
                        st.session_state.requirements_analyzed = False
                        st.rerun()

        # Processar áudio após parar gravação
        if st.session_state.get("processar_audio"):
            audio_array, duracao = st.session_state.gravador.parar()

            # Converter para BytesIO em formato WAV
            audio_buffer = BytesIO()
            sf.write(audio_buffer, audio_array, 16000, format='wav')
            audio_buffer.seek(0)

            st.session_state.ultimo_audio = audio_buffer
            texto, tempo_audio, tempo_transcricao = transcrever_audio(audio_array)
            st.session_state.ultima_transcricao = texto
            st.session_state.tempo_audio = tempo_audio
            st.session_state.tempo_transcricao = tempo_transcricao
            st.session_state.timestamp = datetime.now()

            # # Adiciona ao histórico
            # st.session_state.chatbot_responses.append({
            #     "role": "user",
            #     "content": [{
            #         "type": "audio_file",
            #         "audio_file": audio_buffer,
            #     }]
            # })
            
            # st.session_state.pergunta_isaudio = True
            
            st.session_state["processar_audio"] = False
            st.rerun()

        # # Mostrar transcrição
        # if not st.session_state.gravador.gravando and st.session_state.ultima_transcricao:
        #     st.divider()
        #     st.subheader("📝 Transcrição:")
        #     st.write(st.session_state.ultima_transcricao)

        #     st.caption(f"⏱ **Duração do áudio:** {st.session_state.tempo_audio:.1f}s | "
        #             f"**Tempo de transcrição:** {st.session_state.tempo_transcricao:.1f}s | "
        #             f"**Velocidade:** {st.session_state.tempo_audio/st.session_state.tempo_transcricao:.1f}x")

        #     st.caption(f"🕒 **Timestamp:** {st.session_state.timestamp.strftime('%d/%m/%Y %H:%M:%S')}")
            

    # Mostra histórico antes de gerar nova resposta
    historic()

    # Entrada do usuário
    user_input = st.chat_input("Me pergunte algo.")

    # Se houve entrada manual, prioriza ela; senão, usa transcrição se existir
    if user_input:
        prompt = user_input
        is_audio = False
        st.session_state.isQuestionAudio = False
    elif st.session_state.ultima_transcricao:
        prompt = st.session_state.ultima_transcricao
        # Limpa a transcrição após usar
        st.session_state.ultima_transcricao = ""
        is_audio = True
        st.session_state.isQuestionAudio = True
    else:
        prompt = None
        is_audio = False
        st.session_state.isQuestionAudio = False

    # Se houve nova pergunta
    if prompt:
        if is_audio:
             # Adiciona SOMENTE o áudio ao histórico
            with st.chat_message("user"):
                st.audio(st.session_state.ultimo_audio)
            
            st.session_state.chatbot_responses.append({
                "role": "user",
                "content": [{
                    "type": "audio_file",
                    "audio_file": st.session_state.ultimo_audio,
                }]
            })
            st.session_state.ultimo_audio = None
        else:
            # Mostrar pergunta imediatamente
            with st.chat_message("user"):
                st.write(prompt)
            # Adicionar pergunta no histórico
            st.session_state.chatbot_responses.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

        # Gerar resposta
        with st.spinner("Gerando..."):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp_start = datetime.now()
            # Gerar a resposta do chatbot
            all_uploaded_sentences = [
                s for doc in st.session_state.uploaded_docs for s in doc["sentences"]
            ]

            previous_data = connection_bd.get_previous_questions()
            if previous_data is not None and len(previous_data) > 0:
                previous_questions = [item['question'] for item in previous_data]
                index, similarity = similarity_text.find_similar_question(prompt, previous_questions)
                if similarity > 0.95 and not all_uploaded_sentences:
                    response = previous_data[index]['response']
                else:
                    response = conversation_chatbot(
                        prompt, st.session_state.df, st.session_state.resources,
                        uploaded_sentences=all_uploaded_sentences,
                    )
            else:
                response = conversation_chatbot(
                    prompt, st.session_state.df, st.session_state.resources,
                    uploaded_sentences=all_uploaded_sentences,
                )

        # Checa se a conversão para áudio está ativada
        if on:
            # audio_path = f"script\\output\\audio_output_{st.session_state['user_id']}_{timestamp}.wav"
            audio = texto_para_audio(response)
            st.session_state.chatbot_responses.append({
                    "role": "assistant",
                    "content":[{
                        "type": "audio_file",
                        "audio_file": audio,
                    }]
                })
            st.session_state.isResponseAudio = True
            isResponseAudio = True
        else:
            # Exibir resposta incrementalmente
            full_response = display_incremental_response(response)

            # Adicionar resposta no histórico
            st.session_state.chatbot_responses.append({
                "role": "assistant",
                "content": [{"type": "text", "text": full_response}]
            })
        
        isQuestionAudio = st.session_state.isQuestionAudio
        isResponseAudio = st.session_state.isResponseAudio
        timestamp_end = datetime.now()
        delta = timestamp_end - timestamp_start
        time_in_seconds = delta.total_seconds()
        interaction.log_interaction(prompt, response, isQuestionAudio, isResponseAudio, time_in_seconds)

        st.rerun()

    # ── Seção de Análise de Requisitos ──────────────────────────────────────
    if st.session_state.uploaded_docs:
        st.divider()
        has_reqs = len(st.session_state.extracted_requirements) > 0
        with st.expander("Análise de Requisitos", expanded=has_reqs):
            total_sent = sum(len(d['sentences']) for d in st.session_state.uploaded_docs)
            st.caption(f"{total_sent} sentenças disponíveis nos documentos carregados.")

            if using_ml_model():
                st.success("Classificador ML ativo (treinado no PURE dataset)", icon="🤖")
            else:
                st.warning("Usando classificador baseado em regras — execute a opção [4] no main.py para treinar o modelo ML.", icon="⚙️")

            col_btn, col_clear = st.columns([2, 1])
            with col_btn:
                if st.button("Extrair Requisitos", type="primary", key="btn_extract"):
                    all_sents = [s for d in st.session_state.uploaded_docs for s in d['sentences']]
                    with st.spinner("Classificando sentenças..."):
                        st.session_state.extracted_requirements = extract_requirements(all_sents)
                        st.session_state.user_stories = {}
                        st.session_state.requirements_analyzed = True
                        st.session_state.requirements_grouped = False
                        st.session_state.requirements_duplicates = []
                        st.session_state.requirements_dup_run = False
                    st.rerun()
            with col_clear:
                if has_reqs and st.button("Limpar", key="btn_clear_reqs"):
                    st.session_state.extracted_requirements = []
                    st.session_state.user_stories = {}
                    st.session_state.requirements_analyzed = False
                    st.session_state.requirements_grouped = False
                    st.session_state.requirements_duplicates = []
                    st.session_state.requirements_dup_run = False
                    st.rerun()

            reqs = st.session_state.extracted_requirements
            if not reqs and st.session_state.requirements_analyzed:
                st.info(
                    "Nenhum requisito identificado nas sentenças do documento.\n\n"
                    "**Possíveis causas:**\n"
                    "- O documento não contém linguagem de requisitos (ex: 'deve', 'shall', 'obrigatório')\n"
                    "- O texto extraído está em um formato não reconhecido\n"
                    "- O classificador está em modo regex — treine o modelo ML com a opção **[4]** no `main.py` para maior cobertura",
                    icon="ℹ️",
                )
            if reqs:
                from collections import Counter

                fn_reqs = [r for r in reqs if r['type'] == 'functional']
                nf_reqs = [r for r in reqs if r['type'] == 'non_functional']
                br_reqs = [r for r in reqs if r['type'] == 'business_rule']
                un_reqs = [r for r in reqs if r['type'] == 'uncertain']

                # Contadores de qualidade
                qc = Counter(r['quality_label'] for r in reqs)

                st.markdown(
                    f"**{len(reqs)} requisitos encontrados** — "
                    f"🟢 {len(fn_reqs)} funcionais · "
                    f"🟡 {len(nf_reqs)} não-funcionais · "
                    f"🔵 {len(br_reqs)} regras de negócio"
                    + (f" · ⚪ {len(un_reqs)} incertos" if un_reqs else "")
                )
                st.caption(
                    f"Qualidade: 🟢 {qc.get('Excelente', 0)} Excelente · "
                    f"🟡 {qc.get('Bom', 0)} Bom · "
                    f"🟠 {qc.get('Regular', 0)} Regular · "
                    f"🔴 {qc.get('Ruim', 0)} Ruim"
                )

                tab_tipo, tab_grupo, tab_dup = st.tabs(["Por Tipo", "Por Grupo", "Duplicatas"])

                # ── Aba: Por Tipo ────────────────────────────────────────────
                with tab_tipo:
                    for req_type in ('functional', 'non_functional', 'business_rule', 'uncertain'):
                        typed = [r for r in reqs if r['type'] == req_type]
                        if not typed:
                            continue
                        icon  = TYPE_ICONS[req_type]
                        label = TYPE_LABELS[req_type]
                        with st.expander(f"{icon} {label} ({len(typed)})", expanded=(req_type == 'functional')):
                            for i, req in enumerate(typed):
                                qi = req.get('quality_icon', '')
                                ql = req.get('quality_label', '')
                                st.markdown(f"**{i + 1}.** {req['text']}")
                                issues = req.get('quality_issues', [])
                                if issues:
                                    st.caption(f"{qi} {ql} — _{'; '.join(issues)}_")
                                else:
                                    st.caption(f"{qi} {ql}")

                # ── Aba: Por Grupo ───────────────────────────────────────────
                with tab_grupo:
                    col_grp, _ = st.columns([2, 3])
                    with col_grp:
                        if st.button("Agrupar por Tema", key="btn_group"):
                            with st.spinner("Agrupando requisitos..."):
                                st.session_state.extracted_requirements = group_requirements(reqs)
                                st.session_state.requirements_grouped = True
                            st.rerun()

                    if st.session_state.requirements_grouped and any('group_id' in r for r in reqs):
                        from itertools import groupby
                        sorted_reqs = sorted(reqs, key=lambda r: r.get('group_id', 0))
                        for gid, grp_iter in groupby(sorted_reqs, key=lambda r: r.get('group_id', 0)):
                            grp = list(grp_iter)
                            glabel = grp[0].get('group_label', 'Geral')
                            with st.expander(f"**Grupo {gid + 1} — {glabel}** ({len(grp)} requisitos)", expanded=True):
                                for req in grp:
                                    icon = TYPE_ICONS.get(req['type'], '')
                                    qi   = req.get('quality_icon', '')
                                    st.markdown(f"{icon} {req['text']}")
                                    st.caption(f"{qi} {req.get('quality_label', '')} · {TYPE_LABELS.get(req['type'], req['type'])}")
                    else:
                        st.info("Clique em **Agrupar por Tema** para organizar os requisitos semanticamente (K-means).", icon="💡")

                # ── Aba: Duplicatas ──────────────────────────────────────────
                with tab_dup:
                    col_dup, _ = st.columns([2, 3])
                    with col_dup:
                        if st.button("Detectar Duplicatas", key="btn_dup"):
                            with st.spinner("Calculando similaridade semântica..."):
                                st.session_state.requirements_duplicates = find_duplicates(reqs)
                                st.session_state.requirements_dup_run = True
                            st.rerun()

                    dups = st.session_state.requirements_duplicates
                    if dups:
                        st.warning(f"{len(dups)} par(es) similar(es) encontrado(s) — revise antes de exportar.", icon="⚠️")
                        for pair in dups:
                            sim = pair['similarity']
                            ra  = pair['req_a']
                            rb  = pair['req_b']
                            sim_pct = int(sim * 100)
                            with st.expander(f"Similaridade {sim_pct}% — {ra['text'][:50]}...", expanded=False):
                                st.markdown(f"**A:** {ra['text']}")
                                st.caption(f"{TYPE_ICONS.get(ra['type'],'')} {TYPE_LABELS.get(ra['type'], ra['type'])} · {ra.get('quality_icon','')} {ra.get('quality_label','')}")
                                st.markdown(f"**B:** {rb['text']}")
                                st.caption(f"{TYPE_ICONS.get(rb['type'],'')} {TYPE_LABELS.get(rb['type'], rb['type'])} · {rb.get('quality_icon','')} {rb.get('quality_label','')}")
                                st.progress(sim, text=f"{sim_pct}% similar")
                    elif st.session_state.requirements_dup_run:
                        st.success("Nenhum par de requisitos similar encontrado.", icon="✅")
                    else:
                        st.info("Clique em **Detectar Duplicatas** para identificar requisitos semanticamente similares.", icon="💡")

                # ── Geração de User Stories ──────────────────────────────────
                if fn_reqs:
                    st.divider()
                    st.subheader("User Stories")

                    if st.button("Gerar User Stories", type="primary", key="btn_gen_stories"):
                        stories = {}
                        progress = st.progress(0, text="Iniciando geração...")
                        for i, req in enumerate(fn_reqs):
                            req_idx = reqs.index(req)
                            progress.progress(
                                (i + 1) / len(fn_reqs),
                                text=f"Gerando {i + 1}/{len(fn_reqs)}: {req['text'][:50]}...",
                            )
                            stories[req_idx] = generate_user_story(req['text'])
                        st.session_state.user_stories = stories
                        progress.empty()
                        st.rerun()

                    if st.session_state.user_stories:
                        for req_idx, story in st.session_state.user_stories.items():
                            req = reqs[req_idx]
                            label = req['text'][:70] + ("..." if len(req['text']) > 70 else "")
                            with st.expander(f"US — {label}"):
                                st.markdown(story)
                                st.caption(f"Requisito original: _{req['text']}_")

            # ── Exportar resultados ──────────────────────────────────────────
            if reqs:
                st.divider()
                st.subheader("Exportar")
                col_csv, col_json = st.columns(2)

                with col_csv:
                    df_export = pd.DataFrame([
                        {
                            'requisito':       r['text'],
                            'tipo':            TYPE_LABELS[r['type']],
                            'qualidade':       r.get('quality_label', ''),
                            'score_qualidade': r.get('quality_score', ''),
                            'problemas':       '; '.join(r.get('quality_issues', [])),
                            'grupo':           r.get('group_label', ''),
                        }
                        for r in reqs
                    ])
                    csv_buf = StringIO()
                    df_export.to_csv(csv_buf, index=False, encoding='utf-8')
                    st.download_button(
                        label="Baixar Requisitos (CSV)",
                        data=csv_buf.getvalue().encode('utf-8'),
                        file_name="requisitos.csv",
                        mime="text/csv",
                        key="dl_csv",
                    )

                with col_json:
                    export_data = []
                    for i, req in enumerate(reqs):
                        entry = {
                            'requisito':       req['text'],
                            'tipo':            TYPE_LABELS[req['type']],
                            'qualidade':       req.get('quality_label', ''),
                            'score_qualidade': req.get('quality_score', ''),
                            'problemas':       req.get('quality_issues', []),
                            'grupo':           req.get('group_label', ''),
                        }
                        if i in st.session_state.user_stories:
                            entry['user_story'] = st.session_state.user_stories[i]
                        export_data.append(entry)
                    json_bytes = json.dumps(
                        export_data, ensure_ascii=False, indent=2
                    ).encode('utf-8')
                    st.download_button(
                        label="Baixar Artefatos (JSON)",
                        data=json_bytes,
                        file_name="artefatos.json",
                        mime="application/json",
                        key="dl_json",
                    )


if __name__ == "__main__":
    main()