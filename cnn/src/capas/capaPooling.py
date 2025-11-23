import numpy as np
from src.capas.capaBase import Capa

class CapaMaxPooling2x2(Capa):
    def __init__(self):
        pass

    def forward(self, entrada):
        self.entrada_shape = entrada.shape
        N, C, H, W = entrada.shape
        H_salida = H // 2
        W_salida = W // 2
        salida = np.zeros((N, C, H_salida, W_salida))
        self.mascaras = {}

        for n in range(N):
            for c in range(C):
                for i in range(H_salida):
                    for j in range(W_salida):
                        h0 = i*2
                        w0 = j*2
                        patch = entrada[n, c, h0:h0+2, w0:w0+2]
                        max_val = np.max(patch)
                        salida[n, c, i, j] = max_val
                        self.mascaras[(n,c,i,j)] = (patch == max_val)

        return salida

    def backward(self, gradiente_salida, lr=None):
        N, C, H, W = self.entrada_shape
        H_salida = H // 2
        W_salida = W // 2
        gradiente_entrada = np.zeros(self.entrada_shape)
        for n in range(N):
            for c in range(C):
                for i in range(H_salida):
                    for j in range(W_salida):
                        h0 = i*2
                        w0 = j*2
                        mascara = self.mascaras[(n,c,i,j)]
                        gradiente_entrada[n, c, h0:h0+2, w0:w0+2] += gradiente_salida[n, c, i, j] * mascara
        return gradiente_entrada
