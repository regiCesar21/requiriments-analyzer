import streamlit as st
import time
from datetime import datetime
import threading
import numpy as np
import sounddevice as sd
import whisper
import queue
from gtts import gTTS
import io
import edge_tts
import asyncio

# =======================
# ⚙️ CONFIGURAÇÕES GERAIS
# =======================

FS = 16000
BUFFER_SIZE = 1024

# Escolha o índice correto do seu microfone
MIC_DEVICE_INDEX = None  # None usa o dispositivo padrão

# =======================
# 🔊 CARREGAR MODELO WHISPER
# =======================

@st.cache_resource
def carregar_modelo():
    model = whisper.load_model("small")
    if str(model.device) == 'cpu':
        model.fp16 = False
    return model

model = carregar_modelo()

# ================================
# 🎤 FUNÇÕES DE GRAVAÇÃO
# ================================

class GravadorAudio:
    def __init__(self):
        self.fs = FS
        self.buffer_size = BUFFER_SIZE
        self.device = MIC_DEVICE_INDEX
        self.stream = None
        self.frames = []
        self.gravando = False
        self.queue = queue.Queue()
    
    def callback(self, indata, frames, time_info, status):
        if status:
            print(f"Erro de áudio: {status}")
        if self.gravando:
            self.queue.put(indata.copy())
    
    def iniciar(self):
        if self.gravando:
            return
            
        self.gravando = True
        self.frames = []
        self.queue = queue.Queue()
        
        self.stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype=np.int16,
            blocksize=self.buffer_size,
            callback=self.callback,
            device=self.device
        )
        self.stream.start()
        return time.time()
    
    def parar(self):
        if not self.gravando:
            return np.array([], dtype=np.int16), 0
            
        self.gravando = False
        
        # Aguarda processamento dos últimos buffers
        time.sleep(0.5)
        
        # Coleta todos os frames da fila
        while not self.queue.empty():
            self.frames.append(self.queue.get())
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.frames:
            audio_array = np.concatenate(self.frames, axis=0)
            duracao = len(audio_array) / self.fs
            return audio_array.flatten(), duracao
        else:
            return np.array([], dtype=np.int16), 0

# ================================
# 🎙️ FUNÇÃO DE TRANSCRIÇÃO
# ================================

def transcrever_audio(audio_array):
    if audio_array.size == 0:
        return "", 0, 0

    audio_float = audio_array.astype(np.float32) / 32768.0
    inicio = time.time()
    resultado = model.transcribe(audio_float, fp16=False)
    tempo_transcricao = time.time() - inicio
    texto = resultado["text"].strip()
    tempo_audio = len(audio_array) / FS

    print("📝 Transcrição:", texto)
    print("⏱ Tempo áudio:", tempo_audio, "Tempo transcrição:", tempo_transcricao)
    return texto, tempo_audio, tempo_transcricao


async def texto_para_audio_edge(texto):
    communicate = edge_tts.Communicate(texto, "pt-BR-AntonioNeural")  # Exemplo voz masculina
    audio_fp = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_fp.write(chunk["data"])
    audio_fp.seek(0)
    return audio_fp

# def texto_para_audio(texto):
#     if not texto.strip():
#         return None
    
#     tts = gTTS(text=texto, lang='pt-br')
#     audio_fp = io.BytesIO()
#     tts.write_to_fp(audio_fp)
#     audio_fp.seek(0)
#     return audio_fp

def texto_para_audio(texto):
    return asyncio.run(texto_para_audio_edge(texto))