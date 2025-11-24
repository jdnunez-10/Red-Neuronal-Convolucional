import numpy as np
from src.capas.capaBase import Capa

class CapaConvolucional(Capa):

    def __init__(self, canales_entrada, filtros_salida, tam_kernel=3, stride=1, padding=1):
        self.canales_entrada = canales_entrada
        self.filtros_salida = filtros_salida
        self.tam_kernel = tam_kernel
        self.stride = stride
        self.padding = padding

        # Xavier
        limite = np.sqrt(6 / (canales_entrada * tam_kernel * tam_kernel + filtros_salida * tam_kernel * tam_kernel))
        self.pesos = np.random.uniform(-limite, limite, 
                                       (filtros_salida, canales_entrada, tam_kernel, tam_kernel))
        self.bias = np.zeros(filtros_salida)

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        k, s, p = self.tam_kernel, self.stride, self.padding

        if p > 0:
            X_pad = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        else:
            X_pad = X

        self.X_pad = X_pad

        H_out = (H + 2*p - k) // s + 1
        W_out = (W + 2*p - k) // s + 1

        out = np.zeros((N, self.filtros_salida, H_out, W_out))

        for n in range(N):
            for f in range(self.filtros_salida):
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * s
                        w0 = j * s
                        region = X_pad[n, :, h0:h0+k, w0:w0+k]
                        out[n, f, i, j] = np.sum(region * self.pesos[f]) + self.bias[f]

        return out

    def backward(self, d_out, lr):
        N, C, H, W = self.X.shape
        k, s, p = self.tam_kernel, self.stride, self.padding

        H_out, W_out = d_out.shape[2], d_out.shape[3]

        dW = np.zeros_like(self.pesos)
        dB = np.zeros_like(self.bias)
        dX_pad = np.zeros_like(self.X_pad)

        # --- ROTAR KERNEL 180° ---
        pesos_rotados = np.flip(self.pesos, axis=(2, 3))

        # Gradiente de pesos y entrada
        for n in range(N):
            for f in range(self.filtros_salida):
                dB[f] += np.sum(d_out[n, f])
                for i in range(H_out):
                    for j in range(W_out):
                        h0 = i * s
                        w0 = j * s
                        region = self.X_pad[n, :, h0:h0+k, w0:w0+k]

                        dW[f] += d_out[n, f, i, j] * region
                        dX_pad[n, :, h0:h0+k, w0:w0+k] += d_out[n, f, i, j] * pesos_rotados[f]

        # Quitar padding
        if p > 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad

        # ACTUALIZACIÓN
        self.pesos -= lr * dW
        self.bias  -= lr * dB

        return dX
