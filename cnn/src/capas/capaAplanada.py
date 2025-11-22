import numpy as np
from src.capas.capaBase import Capa

class CapaAplanada(Capa):
    def forward(self, entrada):
        self.entrada_shape = entrada.shape
        return entrada.flatten().reshape(entrada.shape[0], -1)

    def backward(self, gradiente_salida, lr=None):
        return gradiente_salida.reshape(self.entrada_shape)
