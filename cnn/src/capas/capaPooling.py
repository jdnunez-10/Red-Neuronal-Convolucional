import numpy as np
from src.capas.capaBase import Capa

class CapaMaxPooling2x2(Capa):

    def __init__(self):
        self.kernel = 2
        self.stride = 2

    def forward(self, X):

        self.X = X
        N, C, H, W = X.shape

        H_out = H // 2
        W_out = W // 2

        out = np.zeros((N, C, H_out, W_out))
        self.mask = np.zeros_like(X, dtype=bool)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):

                        h0 = i * 2
                        w0 = j * 2

                        region = X[n, c, h0:h0+2, w0:w0+2]
                        max_val = np.max(region)

                        out[n, c, i, j] = max_val

                        # Creamos máscara única para máximo
                        max_mask = (region == max_val)

                        # Si hay dos máximos, elige solo uno para estabilidad
                        if np.sum(max_mask) > 1:
                            # selecciona el primero
                            coords = np.argwhere(max_mask)
                            max_mask[:] = False
                            r, s = coords[0]
                            max_mask[r, s] = True

                        # Guardamos la máscara en posiciones originales
                        self.mask[n, c, h0:h0+2, w0:w0+2] = max_mask

        return out

    def backward(self, d_out, lr=None):

        N, C, H, W = self.X.shape
        dX = np.zeros_like(self.X)

        H_out = d_out.shape[2]
        W_out = d_out.shape[3]

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):

                        h0 = i * 2
                        w0 = j * 2

                        # copiar gradiente SOLO donde la máscara es 1
                        dX[n, c, h0:h0+2, w0:w0+2] += (
                            self.mask[n, c, h0:h0+2, w0:w0+2] * d_out[n, c, i, j]
                        )

        return dX
