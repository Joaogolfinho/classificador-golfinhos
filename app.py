import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

st.set_page_config(page_title="Classificador de Golfinhos", page_icon="🐬")

st.title("🐬 Identificador de Mordidas de Tubarão-Charuto")
st.write("Faça o upload de imagens para verificar e salvar automaticamente no Google Drive.")

# --- CONFIGURAÇÃO DO GOOGLE DRIVE ---
# O código da sua pasta do Projeto já está configurado aqui:
ID_PASTA_PRINCIPAL = "1duAog6h9zZPi6kC9_SLEyVBDK-counWz"

@st.cache_resource
def conectar_drive():
    # Puxa a chave do cofre secreto do Streamlit
    credenciais_dict = json.loads(st.secrets["google_credentials"])
    creds = service_account.Credentials.from_service_account_info(
        credenciais_dict, scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def criar_ou_obter_pasta(nome_pasta, id_pai, servico):
    # Procura se a pasta já existe
    query = f"name='{nome_pasta}' and '{id_pai}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    resultados = servico.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    pastas = resultados.get('files', [])
    
    if pastas:
        return pastas[0].get('id')
    else:
        # Se não existe, cria a pasta
        metadata = {
            'name': nome_pasta,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [id_pai]
        }
        pasta = servico.files().create(body=metadata, fields='id').execute()
        return pasta.get('id')

def subir_para_drive(imagem, nome_arquivo, id_pasta_destino, servico):
    img_bytes = io.BytesIO()
    # Mantém o formato original ou usa JPEG como padrão
    formato = imagem.format if imagem.format else "JPEG"
    imagem.save(img_bytes, format=formato)
    img_bytes.seek(0)
    
    media = MediaIoBaseUpload(img_bytes, mimetype=f'image/{formato.lower()}', resumable=True)
    metadata = {'name': nome_arquivo, 'parents': [id_pasta_destino]}
    
    servico.files().create(body=metadata, media_body=media, fields='id').execute()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_tubarao_charuto.h5')

# Tenta ligar os motores (IA e Google Drive)
try:
    modelo = load_model()
    servico_drive = conectar_drive()
except Exception as e:
    st.error(f"Erro ao carregar o sistema ou conectar ao Drive: {e}")
    st.stop()

arquivos_upload = st.file_uploader("Escolha as imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if arquivos_upload:
    st.info("Processando e enviando para o Google Drive... Por favor, aguarde.")
    
    # Garante que as subpastas existam no Drive
    id_pasta_sem_mordida = criar_ou_obter_pasta("Sem_Mordida", ID_PASTA_PRINCIPAL, servico_drive)
    id_pasta_com_mordida = criar_ou_obter_pasta("Com_Mordida", ID_PASTA_PRINCIPAL, servico_drive)
    
    for arquivo in arquivos_upload:
        st.markdown("---")
        imagem = Image.open(arquivo)
        st.image(imagem, caption=f'Analisando: {arquivo.name}', use_container_width=True)
        
        # Preparar imagem para a IA
        img_redimensionada = imagem.resize((224, 224))
        img_array = np.array(img_redimensionada)
        
        if len(img_array.shape) == 2: 
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 4: 
            img_array = img_array[..., :3]
            
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Previsão
        predicao = modelo.predict(img_array)[0][0]
        
        if predicao > 0.5:
            st.success(f"**Resultado:** Sem Mordida. Enviando para a pasta 'Sem_Mordida' no Drive...")
            subir_para_drive(imagem, arquivo.name, id_pasta_sem_mordida, servico_drive)
        else:
            st.error(f"**Resultado:** ALERTA! Com Mordida. Enviando para a pasta 'Com_Mordida' no Drive...")
            subir_para_drive(imagem, arquivo.name, id_pasta_com_mordida, servico_drive)
            
    st.success("✅ Todas as imagens foram processadas e salvas no seu Google Drive com sucesso!")
