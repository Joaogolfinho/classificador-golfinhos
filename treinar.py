import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Configurações básicas
TAMANHO_IMG = (224, 224)
BATCH_SIZE = 32
DIRETORIO_DADOS = 'dataset'

# 2. Preparação dos dados (com data augmentation para melhorar o modelo)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # Usa 20% para validação
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DIRETORIO_DADOS,
    target_size=TAMANHO_IMG,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DIRETORIO_DADOS,
    target_size=TAMANHO_IMG,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. Criação do Modelo Base (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # Congela os pesos base para treinar apenas o final

# 4. Adicionando as camadas de classificação personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
previsao = Dense(1, activation='sigmoid')(x) # Sigmoid pois são apenas 2 classes

modelo = Model(inputs=base_model.input, outputs=previsao)

# 5. Compilação e Treinamento
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Iniciando o treinamento...")
modelo.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10 # Ajuste o número de épocas conforme necessário
)

# 6. Salvar o modelo treinado
modelo.save('modelo_tubarao_charuto.h5')
print("Modelo salvo como 'modelo_tubarao_charuto.h5'")