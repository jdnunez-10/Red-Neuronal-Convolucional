import numpy as np
from src.activaciones import Softmax

class Predictor:
    """
    Clase para ejecutar predicciones ya entrenado el modelo CNN.
    Incluye el uso de Softmax y el criterio de umbral para detectar
    'ninguna de las anteriores'.
    """

    def __init__(self, modelo):
        self.modelo = modelo
        self.softmax = Softmax()

    def predecir(self, imagen):
        """
        Realiza la predicción de una imagen individual.
        La imagen debe venir con forma (1, canales, alto, ancho)
        """

        # Paso hacia adelante por la red (sin softmax)
        logits = self.modelo.forward(imagen)

        # Aplicar softmax para convertir logits a probabilidades
        probabilidades = self.softmax.forward(logits)[0]

        # Índice de la clase más probable
        indice_pred = int(np.argmax(probabilidades))
        confianza = float(np.max(probabilidades))

        # Aplicar umbral definido en el modelo
        if confianza < self.modelo.umbral:
            return "ninguna", confianza, probabilidades

        return indice_pred, confianza, probabilidades
