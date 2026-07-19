import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile

st.set_page_config(page_title="Classificador de Golfinhos", page_icon="🐬")

st.title("🐬 Identificador de Mordidas de Tubarão-Charuto")
st.write("Analise suas imagens e baixe-as separadas automaticamente.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

modelo = load_model()

arquivos_upload = st.file_uploader("Escolha as imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for arquivo in arquivos_upload:
            imagem = Image.open(arquivo)
            img_redimensionada = imagem.resize((224, 224))
            img_array = np.expand_dims(np.array(img_redimensionada), axis=0) / 255.0
            predicao = modelo.predict(img_array)[0][0]
            
            # Prepara imagem para salvar
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format="JPEG")
            
            if predicao > 0.5:
                zip_file.writestr(f"Sem_Mordida/{arquivo.name}", img_bytes.getvalue())
                st.write(f"✅ {arquivo.name}: Sem Mordida")
            else:
                zip_file.writestr(f"Com_Mordida/{arquivo.name}", img_bytes.getvalue())
                st.write(f"❌ {arquivo.name}: Com Mordida")

    st.download_button("⬇️ Baixar fotos separadas (.zip)", zip_buffer.getvalue(), "resultado.zip", "application/zip")
