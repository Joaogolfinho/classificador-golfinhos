import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

st.set_page_config(page_title="Classificador de Golfinhos", page_icon="🐬")

st.title("🐬 Identificador de Mordidas de Tubarão-Charuto")

# ID da pasta fixa
ID_PASTA_PRINCIPAL = "1duAog6h9zZPi6kC9_SLEyVBDK-counWz"

def get_drive_service():
    # Cria uma nova conexão a cada chamada para evitar erros de sessão
    credenciais_dict = json.loads(st.secrets["google_credentials"])
    creds = service_account.Credentials.from_service_account_info(
        credenciais_dict, scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

modelo = load_model()

arquivos_upload = st.file_uploader("Escolha as imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    if st.button("Processar e enviar para o Drive"):
        servico = get_drive_service()
        
        # Cria pastas apenas uma vez
        for nome_pasta in ["Sem_Mordida", "Com_Mordida"]:
            query = f"name='{nome_pasta}' and '{ID_PASTA_PRINCIPAL}' in parents and trashed=false"
            res = servico.files().list(q=query).execute().get('files', [])
            if not res:
                servico.files().create(body={'name': nome_pasta, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [ID_PASTA_PRINCIPAL]}).execute()
        
        for arquivo in arquivos_upload:
            st.write(f"Processando: {arquivo.name}")
            imagem = Image.open(arquivo)
            img_array = np.expand_dims(np.array(imagem.resize((224, 224))), axis=0) / 255.0
            predicao = modelo.predict(img_array)[0][0]
            
            nome_pasta = "Sem_Mordida" if predicao > 0.5 else "Com_Mordida"
            id_pasta = servico.files().list(q=f"name='{nome_pasta}' and '{ID_PASTA_PRINCIPAL}' in parents").execute().get('files')[0]['id']
            
            # Subir arquivo
            img_bytes = io.BytesIO()
            imagem.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            
            media = MediaIoBaseUpload(img_bytes, mimetype='image/jpeg')
            servico.files().create(body={'name': arquivo.name, 'parents': [id_pasta]}, media_body=media).execute()
            
            st.write(f"✅ {arquivo.name} enviado para {nome_pasta}")
            time.sleep(2) # Pausa maior para garantir estabilidade
            
        st.success("Todas as imagens processadas!")
