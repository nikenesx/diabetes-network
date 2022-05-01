import os

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense

# Отключение использования CUDA ядер
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LOSS_CHARTS_DIR_NAME = 'loss_charts/'
ACCURACY_CHARTS_DIR_NAME = 'accuracy_charts/'
DIABETES_SAVED_MODEL = 'diabetes_model.h5'


def save_accuracy_chart(fitted_model: Sequential, model_history: History, batch_size: int) -> None:
    history = model_history.history
    chart_title = 'Точность'
    train_data_label = 'Обучающая выборка'

    epochs_count = len(history['accuracy'])
    train_data_values = history['accuracy'][1:]

    epochs = range(1, epochs_count)
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    axes.plot(epochs, train_data_values, label=train_data_label, color='#03bcff')

    axes.set_title(chart_title)
    axes.set_xlabel('Эпохи')
    axes.set_ylabel(chart_title)
    axes.legend()

    if not os.path.exists(ACCURACY_CHARTS_DIR_NAME):
        os.mkdir(ACCURACY_CHARTS_DIR_NAME)

    neurons_count = '-'.join([str(layer.output.shape[1]) for layer in fitted_model.layers])
    figure.savefig(f'{ACCURACY_CHARTS_DIR_NAME}/{neurons_count}_{batch_size}_{epochs_count}_accuracy.jpg')

    plt.close()


def save_loss_chart(fitted_model: Sequential, model_history: History, batch_size: int) -> None:
    """
    Сохраняет изображения с графиком потерь.

    :param fitted_model: обученная модель
    :param model_history: история обучения модели
    """
    history = model_history.history
    chart_title = 'Потери'
    train_data_label = 'Обучающая выборка'
    validation_data_label = 'Выборка валидации'

    epochs_count = len(history['loss'])
    train_data_values = history['loss'][1:]
    validation__data_values = history['val_loss'][1:]

    epochs = range(1, epochs_count)
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    axes.plot(epochs, train_data_values, label=train_data_label, color='#03bcff')
    axes.plot(epochs, validation__data_values, label=validation_data_label, color='#e06704')

    axes.set_title(chart_title)
    axes.set_xlabel('Эпохи')
    axes.set_ylabel(chart_title)
    axes.legend()

    if not os.path.exists(LOSS_CHARTS_DIR_NAME):
        os.mkdir(LOSS_CHARTS_DIR_NAME)

    neurons_count = '-'.join([str(layer.output.shape[1]) for layer in fitted_model.layers])
    figure.savefig(f'{LOSS_CHARTS_DIR_NAME}/{neurons_count}_{batch_size}_{epochs_count}_loss.jpg')

    plt.close()


def train_diabetes_name(saved_model_name: str, show_stat: bool = False) -> None:
    """
    Получает датасет обучает модель для предсказания развития диабета.

    :param saved_model_name: название обученной модели
    :param show_stat: сохраняет графики точности и потерь, если True
    """
    dataset = np.loadtxt('datasets/diabetes_dataset.csv', delimiter=',')
    input_vectors = dataset[:, 0:8]
    result_values = dataset[:, 8]

    model = Sequential()

    model.add(Dense(8, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    batch_size = 10
    history = model.fit(input_vectors, result_values, epochs=200, batch_size=batch_size, validation_split=0.2)
    model.save(saved_model_name)

    if show_stat:
        save_loss_chart(fitted_model=model, model_history=history, batch_size=batch_size)
        save_accuracy_chart(fitted_model=model, model_history=history, batch_size=batch_size)

        accuracy = model.evaluate(input_vectors, result_values)
        accuracy_value = round(float(accuracy[1] * 100), 2)
        print(f'Точность: {accuracy_value}%')


if __name__ == '__main__':
    model_name = DIABETES_SAVED_MODEL
    train_diabetes_name(saved_model_name=model_name, show_stat=True)
