import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import zipfile

st.set_page_config(page_title="Classificador de Golfinhos", page_icon="🐬", layout="wide")

st.title("🐬 Identificador de Mordidas de Tubarão-Charuto")
st.write("Analise as imagens, verifique a predição e baixe o resultado organizado.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

modelo = load_model()

arquivos_upload = st.file_uploader("Escolha as fotos para análise", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    st.info(f"Processando {len(arquivos_upload)} imagens...")
    
    # Prepara o arquivo ZIP na memória
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for arquivo in arquivos_upload:
            st.markdown("---")
            # Abre e exibe a imagem
            imagem = Image.open(arquivo)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(imagem, use_container_width=True)
            
            # Processamento da IA
            img_redimensionada = imagem.resize((224, 224))
            img_array = np.array(img_redimensionada)
            
            # Ajuste de cores/canais
            if len(img_array.shape) == 2: 
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] == 4: 
                img_array = img_array[..., :3]
            
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predicao = modelo.predict(img_array)[0][0]
            
            # Resultado visual na Coluna 2
            with col2:
                if predicao > 0.5:
                    st.success(f"**Resultado:** Sem Mordida. (Confiança: {predicao*100:.2f}%)")
                    pasta = "Sem_Mordida"
                else:
                    st.error(f"**Resultado:** ALERTA! Com Mordida. (Confiança: {(1-predicao)*100:.2f}%)")
                    pasta = "Com_Mordida"
            
            # Salva no ZIP na pasta correta dentro da memória
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format=imagem.format if imagem.format else "JPEG")
            zip_file.writestr(f"{pasta}/{arquivo.name}", img_bytes.getvalue())

    # Botão de download no final
    st.markdown("---")
    st.download_button(
        label="⬇️ Baixar todas as fotos separadas (.zip)",
        data=zip_buffer.getvalue(),
        file_name="resultado_analise.zip",
        mime="application/zip"
    )
    st.success("Processamento concluído!")
