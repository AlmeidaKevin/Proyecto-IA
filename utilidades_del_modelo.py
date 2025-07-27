import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import matplotlib.pyplot as plt

def cargar_datos():
    (x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = mnist.load_data()
    return (x_entrenamiento / 255.0, y_entrenamiento), (x_prueba / 255.0, y_prueba)

def crear_modelo():
    modelo = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    modelo.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    return modelo

def entrenar_modelo(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    historial = modelo.fit(x_entrenamiento, y_entrenamiento, epochs=5, validation_data=(x_prueba, y_prueba))
    return modelo, historial

def graficar_metricas(historial):
    plt.plot(historial.history['accuracy'], label='Precisión')
    plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
    plt.legend()
    plt.show()
