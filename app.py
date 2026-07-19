import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile
import base64

# Configuração da página
st.set_page_config(page_title="Projeto Golfinho Rotador", page_icon="🐬", layout="centered")

# --- CSS PARA DESIGN CENTRALIZADO E FUNDO ---
def set_bg_and_style(image_file):
    with open(image_file, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{img_data});
            background-size: cover;
        }}
        .centered-title {{ text-align: center; color: white; }}
        </style>
    """, unsafe_allow_html=True)

# Aplica o fundo com o nome exato do seu arquivo
set_bg_and_style("fundo.jpg.tif")

# --- CABEÇALHO CENTRALIZADO ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    # Ajustado para o nome exato da logo
    st.image("fundo.png.png", width=100)
with col_titulo:
    st.markdown("<h1 class='centered-title'>Identificador de Mordidas de Tubarão-Charuto em Golfinhos e Tartarugas - Fernando de Noronha</h1>", unsafe_allow_html=True)

st.markdown("<h3 class='centered-title'>Analisador de imagens para atividades de pesquisa - Projeto Golfinho Rotador</h3>", unsafe_allow_html=True)

# --- RESTO DO CÓDIGO (IA + ZIP) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

modelo = load_model()

st.markdown("---")
arquivos_upload = st.file_uploader("Escolha as fotos para análise", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    st.info(f"Processando {len(arquivos_upload)} imagens...")
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for arquivo in arquivos_upload:
            st.markdown("---")
            imagem = Image.open(arquivo)
            # Exibe imagem centralizada
            st.image(imagem, width=300)
            
            img_array = np.expand_dims(np.array(imagem.resize((224, 224))), axis=0) / 255.0
            predicao = modelo.predict(img_array)[0][0]
            
            if predicao > 0.5:
                st.success(f"Resultado: Sem Mordida")
                pasta = "Sem_Mordida"
            else:
                st.error(f"Resultado: ALERTA! Com Mordida")
                pasta = "Com_Mordida"
            
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format="JPEG")
            zip_file.writestr(f"{pasta}/{arquivo.name}", img_bytes.getvalue())

    st.markdown("---")
    st.download_button("⬇️ Baixar fotos organizadas (.zip)", zip_buffer.getvalue(), "resultado_analise.zip", "application/zip")
