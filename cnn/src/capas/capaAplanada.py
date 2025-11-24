
import numpy as np
from src.capas.capaBase import Capa

class CapaAplanada(Capa):
    def forward(self, entrada):
        # Guardar entrada y su forma original
        self.entrada = entrada
        self.entrada_shape = entrada.shape
        # Aplanar manteniendo el batch
        return entrada.reshape(entrada.shape[0], -1)

    def backward(self, gradiente_salida, lr=None):
        # Restaurar la forma original
        return gradiente_salida.reshape(self.entrada_shape)