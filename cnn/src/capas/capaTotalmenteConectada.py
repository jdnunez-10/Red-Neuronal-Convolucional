import numpy as np
from src.capas.capaBase import Capa

class CapaTotalmenteConectada(Capa):
    def __init__(self, tamaño_entrada, tamaño_salida):
        self.pesos = np.random.randn(tamaño_entrada, tamaño_salida) * 0.01
        self.bias = np.zeros(tamaño_salida)

    def forward(self, entrada):
        self.entrada = entrada
        return np.dot(entrada, self.pesos) + self.bias

    def backward(self, gradiente_salida, lr=1e-3):
        gradiente_pesos = np.dot(self.entrada.T, gradiente_salida)
        gradiente_bias = np.sum(gradiente_salida, axis=0)
        gradiente_entrada = np.dot(gradiente_salida, self.pesos.T)

        # actualizar
        self.pesos -= lr * gradiente_pesos
        self.bias -= lr * gradiente_bias

        return gradiente_entrada
