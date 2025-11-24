
import numpy as np
from src.capas.capaBase import Capa

class ReLU(Capa):
    def forward(self, x):
        self.entrada = x
        return np.maximum(0, x)

    def backward(self, gradiente_salida, lr=None):
        return gradiente_salida * (self.entrada > 0)


class Softmax(Capa):
    def forward(self, x):
        # estabilidad numérica
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.salida = exp / np.sum(exp, axis=1, keepdims=True)
        return self.salida

    def backward(self, y_true):
        """
        Gradiente simplificado cuando se usa con Cross-Entropy.
        y_true debe ser one-hot.
        """
        N = y_true.shape[0]
        return (self.salida - y_true) / N