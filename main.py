# Отключим предупреждения в колабе. Будет меньше лишней информации в выводе
import warnings
warnings.simplefilter("ignore")

# Библиотеки для загрузки данных
import requests
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import gdown
import time
import glob
from keras.utils import get_file

# Библиотеки для работы с графиками
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(42)

# Библиотеки для подготовки данных
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
)

# Методы из библиотек для работы с моделью
from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout,
    add
)
from keras.regularizers import l1, l2, l1_l2
from keras.utils import (
    to_categorical,
    plot_model,
    image_dataset_from_directory,
    load_img,
    img_to_array
)
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import BinaryAccuracy, CategoricalAccuracy
from keras.optimizers import Adam, RMSprop
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError

import streamlit as st

from sklearn.utils.class_weight import compute_class_weight

from PIL import Image
import io

st.set_page_config(page_title = "Handwritten Letters Dataset")
st.title("Handwritten Letters Dataset")
st.write(
    """
    This app visualizes data from the A-Z Handwritten Letters Dataset .
    It showcases grayscale images of handwritten letters from A to Z and allows you to explore how different characters appear
    in the dataset. Just use the widgets below to browse through the samples!
    """
)

file_id = '1F2BSo9VarkOo_ahkKfeS4KiGYN7QRYAZ'
download_url = f'https://drive.google.com/uc?id={file_id}'

dataset = np.loadtxt(
    download_url,
    delimiter = ','
)

X = dataset[:, 1:785]
Y = dataset[:, 0]

(x_temp, x_test, y_temp, y_test) = train_test_split(
    X, Y,
    test_size = 0.2,
    shuffle = True
)

(x_train, x_val, y_train, y_val) = train_test_split(
    x_temp, y_temp,
    test_size = 0.25,
    shuffle = True
)

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')

word_dict = {
    0:'A', 1:'B',
    2:'C', 3:'D',
    4:'E', 5:'F',
    6:'G', 7:'H',
    8:'I', 9:'J',
    10:'K', 11:'L',
    12:'M', 13:'N',
    14:'O', 15:'P',
    16:'Q', 17:'R',
    18:'S', 19:'T',
    20:'U', 21:'V',
    22:'W', 23:'X',
    24:'Y', 25:'Z'
}
st.session_state.word_dict = word_dict

st.session_state.x_train = x_train.astype('float32') / 255
st.session_state.x_test = x_test.astype('float32') / 255
st.session_state.x_val = x_val.astype('float32') / 255

class_weights = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train),
    y = y_train
)

# Преобразуем в словарь
st.session_state['class_weights'] = dict(zip(np.unique(y_train), class_weights))

num_classes = pd.Series(y_train).nunique()

st.session_state.y_train = to_categorical(y_train, num_classes) # Кодируем обучающие метки на 26 классов
st.session_state.y_test = to_categorical(y_test, num_classes)   # Кодируем тестовые метки на 26 классов
st.session_state.y_val = to_categorical(y_val, num_classes)   # Кодируем тестовые метки на 26 классов

st.session_state['x_train.shape[1]'] = x_train.shape[1]

def show_confusion_matrix(y_true, y_pred, class_labels):
    # Матрица ошибок
    cm = confusion_matrix(
		y_true,
        y_pred,
        normalize='true'
    )

    # Округление значений матрицы ошибок
    cm = np.around(cm, 2)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(f'Матрица ошибок', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    disp.plot(ax=ax, cmap='Blues', values_format='.2f')  # только допустимые параметры

    # Явно задаём тики и подписи
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_yticklabels(class_labels)
    
    plt.gca().images[-1].colorbar.remove()  # Убираем ненужную цветовую шкалу
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси
    st.pyplot(fig)

    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    st.write('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))

st.header("Настройка нейросети")
st.sidebar.header("Настройка архитектуры нейросети")
st.session_state['history'] = []
st.session_state['trained'] = False

def build_model():
    layer_number = st.radio(
        'Сколько скрытых слоёв вы хотите?',
        [1, 2, 3, 4, 5]
    )

    input_layer = Input(shape = (st.session_state['x_train.shape[1]'],))
    x = input_layer

    # Ввод параметров для каждого слоя
    for i in range(layer_number):
        neurons = st.sidebar.slider(
            f'Количество нейронов в слое {i + 1}', 
            min_value = 16, max_value = 512, value = 64, step = 4
        )

        x = Dense(neurons, activation = 'relu')(x)

    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)
    
    model.compile(
        optimizer = RMSprop(learning_rate = 5e-4),
        loss = CategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    return model

model = build_model()
st.session_state['model'] = model
st.session_state['model'].summary()
plot_model(
	st.session_state['model'],
	to_file = 'model_plot.png',
	show_shapes = True,
	show_layer_names = True,
	show_layer_activations = True,
    dpi = 80
)
st.image('model_plot.png', caption='Архитектура модели')

col1, col2 = st.columns(2)

with col1:
    epochs = st.slider('Сколько эпох обучать модель?', 1, 50, 10)

with col2:
    batch_size = st.selectbox(
        'Какой размер одного батча?',
        options = [32, 64, 128, 256, 512],
        index = 2
    )

if st.session_state['model'] and st.button("Обучить модель"):
    with st.spinner("Обучение модели..."):
        history = model.fit(
            st.session_state.x_train, st.session_state.y_train,
            validation_data = (st.session_state.x_val, st.session_state.y_val),
            epochs = epochs,
            batch_size = batch_size,
            verbose = 1,
            class_weight = st.session_state['class_weights']
        )
        st.session_state['model'] = model
        st.success("Модель обучена!")
        st.session_state['trained'] = True

        test_loss, test_acc = st.session_state['model'].evaluate(st.session_state.x_test, st.session_state.y_test)
        st.write(f'Точность на тестовом образце: {test_acc:.4f}')
        st.write(f'Потери на тестовом образце: {test_loss:.4f}')

        predictions = st.session_state['model'].predict(st.session_state.x_test)
        y_pred = np.argmax(predictions, axis = 1)
        y_test_labels = np.argmax(st.session_state.y_test, axis=1)

        show_confusion_matrix(y_test_labels, y_pred, [word_dict[i] for i in range(26)])
        report = classification_report(y_test_labels, y_pred)
        st.text('Отчет о классификации:\n')
        st.text(report)
     
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # График точности
        axes[0].plot(history.history['accuracy'], label='Обучающая выборка')
        axes[0].plot(history.history['val_accuracy'], label='Проверочная выборка')
        axes[0].set_title('График точности')
        axes[0].set_xlabel('Эпохи')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)
        
        # График потерь
        axes[1].plot(history.history['loss'], label='Обучающая выборка')
        axes[1].plot(history.history['val_loss'], label='Проверочная выборка')
        axes[1].set_title('График потерь')
        axes[1].set_xlabel('Эпохи')
        axes[1].set_ylabel('Потери')
        axes[1].legend()
        axes[1].grid(True)
        
        # Просто выводим график — без сохранения в session_state
        st.pyplot(fig)

if st.session_state['model']:
    uploaded_file = st.file_uploader("Загрузите изображение рукописной буквы", type = ["png", "jpg"])

    if uploaded_file is not None and st.button("Сделать прогнозирование"):
        image = Image.open(io.BytesIO(uploaded_file.read())).convert('L')  # Конвертация в grayscale
        image = image.resize((28, 28))  # Изменение размера до 28x28
        image_array = np.array(image).astype('float32') / 255.0  # Нормализация
        image_array = image_array.flatten()  # Преобразование в одномерный массив
        image_array = np.expand_dims(image_array, axis=0)  # Добавление размерности batch
        
        prediction = st.session_state['model'].predict(image_array)
        predicted_class = np.argmax(prediction, axis = 1)[0]
        
        st.write(f"Предсказанный класс: {st.session_state.word_dict[predicted_class]}")
        st.image(image, caption = 'Загруженное изображение', width=150)          
