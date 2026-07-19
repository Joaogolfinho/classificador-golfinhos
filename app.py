import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Classificador de Golfinhos", page_icon="🐬")

st.title("🐬 Identificador de Mordidas de Tubarão-Charuto")
st.write("Faça o upload de imagens para verificar se há marcas de mordida.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

try:
    modelo = load_model()
except Exception as e:
    st.error("Modelo não encontrado.")
    st.stop()

arquivos_upload = st.file_uploader("Escolha as imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    for arquivo in arquivos_upload:
        st.markdown("---")
        imagem = Image.open(arquivo)
        st.image(imagem, caption=f'Arquivo: {arquivo.name}', use_container_width=True)
        
        img_redimensionada = imagem.resize((224, 224))
        img_array = np.array(img_redimensionada)
        
        if len(img_array.shape) == 2: 
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4: 
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predicao = modelo.predict(img_array)[0][0]
        
        if predicao > 0.5:
            st.success(f"**Resultado:** Sem Mordida. (Confiança: {predicao*100:.2f}%)")
        else:
            st.error(f"**Resultado:** ALERTA! Com Mordida. (Confiança: {(1-predicao)*100:.2f}%)")
