import numpy as np

class ReLU:
    def forward(self, x):
        self.entrada = x
        return np.maximum(0, x)

    def backward(self, gradiente_salida, lr=None):
        gradiente = gradiente_salida.copy()
        gradiente[self.entrada <= 0] = 0
        return gradiente

class Softmax:
    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

