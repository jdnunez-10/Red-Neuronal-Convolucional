import numpy as np
from src.activaciones import Softmax

class Predictor:
    """
    Clase para ejecutar predicciones ya entrenado el modelo CNN.
    Incluye el uso de Softmax y el criterio de umbral para detectar
    'ninguna de las anteriores'.
    """

    def __init__(self, modelo, clases=None):
        self.modelo = modelo
        self.softmax = Softmax()
        self.clases = clases  # lista de nombres de clases opcional

    def predecir(self, imagen):
        """
        Predicción de una imagen individual.
        La imagen debe tener forma (1, canales, alto, ancho).
        """
        logits = self.modelo.forward(imagen)
        probabilidades = self.softmax.forward(logits)[0]

        indice_pred = int(np.argmax(probabilidades))
        confianza = float(np.max(probabilidades))

        if confianza < self.modelo.umbral:
            return "ninguna", confianza, probabilidades

        if self.clases:
            return self.clases[indice_pred], confianza, probabilidades
        return indice_pred, confianza, probabilidades

    def predecir_batch(self, X):
        """
        Predicción de un batch de imágenes.
        X debe tener forma (N, canales, alto, ancho).
        """
        logits = self.modelo.forward(X)
        probabilidades = self.softmax.forward(logits)

        pred_indices = np.argmax(probabilidades, axis=1)
        confianzas = np.max(probabilidades, axis=1)

        resultados = []
        for i, (idx, conf) in enumerate(zip(pred_indices, confianzas)):
            if conf < self.modelo.umbral:
                resultados.append(("ninguna", conf, probabilidades[i]))
            else:
                if self.clases:
                    resultados.append((self.clases[idx], conf, probabilidades[i]))
                else:
                    resultados.append((idx, conf, probabilidades[i]))
        return resultados
