import numpy as np
from src.capas.capaBase import Capa

class CapaTotalmenteConectada(Capa):
    def __init__(self, tamaño_entrada, tamaño_salida):
        limite = np.sqrt(6.0 / (tamaño_entrada + tamaño_salida))
        self.pesos = np.random.uniform(-limite, limite, (tamaño_entrada, tamaño_salida))
        self.bias = np.zeros(tamaño_salida)

    def forward(self, entrada):
        self.entrada = entrada
        return np.dot(entrada, self.pesos) + self.bias

    def backward(self, gradiente_salida, lr):
        N = self.entrada.shape[0]

        # Gradientes
        d_pesos = np.dot(self.entrada.T, gradiente_salida) / N
        d_bias = np.sum(gradiente_salida, axis=0) / N

        # Gradiente hacia la entrada
        grad_entrada = np.dot(gradiente_salida, self.pesos.T)

        # Actualizar parámetros
        self.pesos -= lr * d_pesos
        self.bias -= lr * d_bias

        return grad_entrada
