import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile
import base64

# Configuração da página
st.set_page_config(page_title="Projeto Golfinho Rotador", page_icon="🐬", layout="centered")

# --- CSS PARA DESIGN CENTRALIZADO ---
def set_bg_and_style(image_file):
    try:
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
            .main-title {{ text-align: center; color: white; font-size: 2.2em; }}
            .sub-title {{ text-align: center; color: #f0f0f0; }}
            .logo-container {{ display: flex; justify-content: center; margin-bottom: 20px; }}
            </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Erro ao carregar fundo: {e}")

# Nome do arquivo de fundo simplificado
set_bg_and_style("fundo.jpg")

# --- CABEÇALHO CENTRALIZADO ---
st.markdown("<div class='logo-container'>", unsafe_allow_html=True)
# Nome do arquivo da logo simplificado
st.image("logo.png", width=180)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Identificador de Mordidas de Tubarão-Charuto em Golfinhos - Fernando de Noronha</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-title'>Analisador de imagens para atividades de pesquisa - Projeto Golfinho Rotador</h3>", unsafe_allow_html=True)

# --- RESTO DO CÓDIGO ---
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
            imagem = Image.open(arquivo)
            st.image(imagem, width=300)
            
            img_array = np.expand_dims(np.array(imagem.resize((224, 224))), axis=0) / 255.0
            predicao = modelo.predict(img_array)[0][0]
            
            if predicao > 0.5:
                st.success("Resultado: Sem Mordida")
                pasta = "Sem_Mordida"
            else:
                st.error("Resultado: ALERTA! Com Mordida")
                pasta = "Com_Mordida"
            
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format="JPEG")
            zip_file.writestr(f"{pasta}/{arquivo.name}", img_bytes.getvalue())

    st.markdown("---")
    st.download_button("⬇️ Baixar fotos organizadas (.zip)", zip_buffer.getvalue(), "resultado_analise.zip", "application/zip")
