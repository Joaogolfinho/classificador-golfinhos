import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile
import base64
import os

st.set_page_config(page_title="Projeto Golfinho Rotador", page_icon="🐬", layout="centered")

# --- CSS E ESTILOS ---
def set_bg_and_style(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url(data:image/jpeg;base64,{img_data});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .header-container {{ display: flex; align-items: center; justify-content: center; gap: 20px; }}
            .main-title {{ color: white; font-size: 2.0em; margin: 0; }}
            .sub-title {{ text-align: center; color: #f0f0f0; margin-top: 10px; }}
            </style>
        """, unsafe_allow_html=True)

set_bg_and_style("fundo.jpg")

# Cabeçalho
st.markdown("<div class='header-container'>", unsafe_allow_html=True)
if os.path.exists("logo.png"):
    st.image("logo.png", width=120)
st.markdown("<h1 class='main-title'>Identificador de Mordidas de<br>Tubarão-Charuto em Golfinhos -<br>Fernando de Noronha</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>Analisador de imagens para pesquisa - Projeto Golfinho Rotador</h3>", unsafe_allow_html=True)

# --- IA ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

modelo = load_model()

st.markdown("---")
arquivos_upload = st.file_uploader("Escolha as fotos para análise", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for arquivo in arquivos_upload:
            st.markdown("---")
            imagem = Image.open(arquivo).convert("RGB")
            st.image(imagem, width=300)
            
            img_array = np.expand_dims(np.array(imagem.resize((224, 224))), axis=0) / 255.0
            predicao = modelo.predict(img_array)[0][0]
            
            # Cálculo da confiança
            confianca = predicao if predicao > 0.5 else (1 - predicao)
            
            # Lógica de Classificação
            if confianca < 0.80:
                st.warning(f"⚠️ Não tenho certeza (Confiança: {confianca*100:.2f}%)")
                pasta = "Nao_Tenho_Certeza"
            elif predicao > 0.5:
                st.success(f"✅ Sem Mordida (Confiança: {confianca*100:.2f}%)")
                pasta = "Sem_Mordida"
            else:
                st.error(f"❌ Com Mordida (Confiança: {confianca*100:.2f}%)")
                pasta = "Com_Mordida"
            
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format="JPEG")
            zip_file.writestr(f"{pasta}/{arquivo.name}", img_bytes.getvalue())

    st.markdown("---")
    st.download_button("⬇️ Baixar fotos organizadas (.zip)", zip_buffer.getvalue(), "resultado_analise.zip", "application/zip")
