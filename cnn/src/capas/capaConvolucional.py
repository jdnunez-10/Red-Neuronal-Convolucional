import numpy as np
from src.capas.capaBase import Capa

class CapaConvolucional(Capa):
    def __init__(self, canales_entrada, filtros_salida, tam_kernel=5, stride=1, padding=2):
        self.canales_entrada = canales_entrada
        self.filtros_salida = filtros_salida
        self.tam_kernel = tam_kernel
        self.stride = stride
        self.padding = padding

        limite = np.sqrt(6.0 / (canales_entrada*tam_kernel*tam_kernel + filtros_salida*tam_kernel*tam_kernel))
        self.pesos = np.random.uniform(-limite, limite, (filtros_salida, canales_entrada, tam_kernel, tam_kernel))
        self.bias = np.zeros((filtros_salida, 1))

    def forward(self, entrada):
        self.entrada = entrada
        N, C, H, W = entrada.shape
        k, s, p = self.tam_kernel, self.stride, self.padding
        H_salida = (H - k + 2*p)//s + 1
        W_salida = (W - k + 2*p)//s + 1

        if p > 0:
            entrada_pad = np.pad(entrada, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            entrada_pad = entrada

        salida = np.zeros((N, self.filtros_salida, H_salida, W_salida))

        for n in range(N):
            for f in range(self.filtros_salida):
                for i in range(H_salida):
                    for j in range(W_salida):
                        h0 = i*s
                        w0 = j*s
                        region = entrada_pad[n, :, h0:h0+k, w0:w0+k]
                        salida[n, f, i, j] = np.sum(region * self.pesos[f]) + self.bias[f]

        return salida

    def backward(self, gradiente_salida, lr=1e-3):
        entrada = self.entrada
        N, C, H, W = entrada.shape
        k, s, p = self.tam_kernel, self.stride, self.padding

        if p > 0:
            entrada_pad = np.pad(entrada, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            entrada_pad = entrada

        H_salida = gradiente_salida.shape[2]
        W_salida = gradiente_salida.shape[3]
        d_entrada_pad = np.zeros_like(entrada_pad)
        d_pesos = np.zeros_like(self.pesos)
        d_bias = np.zeros_like(self.bias)

        for n in range(N):
            for f in range(self.filtros_salida):
                for i in range(H_salida):
                    for j in range(W_salida):
                        h0 = i*s
                        w0 = j*s
                        region = entrada_pad[n, :, h0:h0+k, w0:w0+k]
                        d_pesos[f] += gradiente_salida[n, f, i, j] * region
                        d_bias[f] += gradiente_salida[n, f, i, j]
                        d_entrada_pad[n, :, h0:h0+k, w0:w0+k] += gradiente_salida[n, f, i, j] * self.pesos[f]

        if p > 0:
            d_entrada = d_entrada_pad[:, :, p:-p, p:-p]
        else:
            d_entrada = d_entrada_pad

        # actualizar parámetros
        self.pesos -= lr * (d_pesos / N)
        self.bias -= lr * (d_bias / N)

        return d_entrada
