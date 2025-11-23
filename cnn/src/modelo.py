import numpy as np
import pickle
from src.capas.capaConvolucional import CapaConvolucional
from src.capas.capaPooling import CapaMaxPooling2x2
from src.capas.capaAplanada import CapaAplanada
from src.capas.capaTotalmenteConectada import CapaTotalmenteConectada
from src.activaciones import ReLU, Softmax

class CNNMinima:
    """
    Red Neuronal Convolucional mínima programada desde cero.
    Arquitectura:
        Conv → ReLU → Pool
        Conv → ReLU → Pool
        Flatten
        FC → ReLU
        FC → Softmax
    """

    def __init__(self, tamaño_entrada=(1,64,64), num_clases=5, lr=1e-3, umbral=0.6):
        C, H, W = tamaño_entrada
        self.lr = lr
        self.umbral = umbral
        self.num_clases = num_clases

        print(f"Inicializando CNN con entrada: {tamaño_entrada}")

        # ------- CAPA 1 -------
        self.conv1 = CapaConvolucional(C, 4, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = CapaMaxPooling2x2()

        # ------- CAPA 2 -------
        self.conv2 = CapaConvolucional(4, 8, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = CapaMaxPooling2x2()

        # ------- CAPAS FINALES -------
        self.flatten = CapaAplanada()

        # Calcular tamaño del vector aplanado CORRECTAMENTE
        print(f"Calculando tamaño del vector aplanado...")
        dummy = np.zeros((1, C, H, W))
        
        # Pasar por cada capa paso a paso
        x = self.conv1.forward(dummy)
        print(f"  Después Conv1: {x.shape}")
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        print(f"  Después Pool1: {x.shape}")
        
        x = self.conv2.forward(x)
        print(f"  Después Conv2: {x.shape}")
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        print(f"  Después Pool2: {x.shape}")
        
        x = self.flatten.forward(x)
        tamaño_flat = x.shape[1]

        print(f"  Tamaño final aplanado: {tamaño_flat}")

        # Capa totalmente conectada
        self.fc1 = CapaTotalmenteConectada(tamaño_flat, 64)
        self.relu_fc = ReLU()

        # Capa de salida
        self.fc2 = CapaTotalmenteConectada(64, num_clases)
        self.softmax = Softmax()
        
        print(f"✓ CNN inicializada correctamente\n")

    # --------------------------------------------------
    #                   FORWARD
    # --------------------------------------------------
    def forward(self, X):
        """
        Propagación hacia adelante a través de toda la red.
        X: entrada con forma (N, C, H, W)
        Retorna: logits (sin softmax) con forma (N, num_clases)
        """
        x = self.conv1.forward(X)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)

        x = self.fc1.forward(x)
        x = self.relu_fc.forward(x)

        logits = self.fc2.forward(x)
        return logits

    # --------------------------------------------------
    #                 BACKWARD
    # --------------------------------------------------
    def backward(self, gradiente, lr):
        """
        Backpropagation a través de toda la red.
        """
        grad = self.fc2.backward(gradiente, lr)
        grad = self.relu_fc.backward(grad)
        grad = self.fc1.backward(grad, lr)
        
        grad = self.flatten.backward(grad)
        
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)
        
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        
        return grad

    # --------------------------------------------------
    #                 PREDICCIÓN
    # --------------------------------------------------
    def predecir_con_umbral(self, X):
        """
        Devuelve la clase predicha o 'ninguna' si la probabilidad máxima
        no supera el umbral definido.
        """
        logits = self.forward(X)
        probs = self.softmax.forward(logits)

        pred_indices = np.argmax(probs, axis=1)
        pred_vals = np.max(probs, axis=1)

        etiquetas_finales = []
        for p, m in zip(pred_indices, pred_vals):
            if m < self.umbral:
                etiquetas_finales.append("ninguna")
            else:
                etiquetas_finales.append(int(p))

        return etiquetas_finales, probs

    # --------------------------------------------------
    #            GUARDAR Y CARGAR PESOS
    # --------------------------------------------------
    def guardar_pesos(self, ruta):
        """
        Guarda los pesos del modelo en un archivo pickle.
        """
        pesos = {
            'conv1_pesos': self.conv1.pesos,
            'conv1_bias': self.conv1.bias,
            'conv2_pesos': self.conv2.pesos,
            'conv2_bias': self.conv2.bias,
            'fc1_pesos': self.fc1.pesos,
            'fc1_bias': self.fc1.bias,
            'fc2_pesos': self.fc2.pesos,
            'fc2_bias': self.fc2.bias,
        }
        with open(ruta, 'wb') as f:
            pickle.dump(pesos, f)
        print(f"✓ Pesos guardados en {ruta}")

    def cargar_pesos(self, ruta):
        """
        Carga los pesos del modelo desde un archivo pickle.
        """
        with open(ruta, 'rb') as f:
            pesos = pickle.load(f)
        
        self.conv1.pesos = pesos['conv1_pesos']
        self.conv1.bias = pesos['conv1_bias']
        self.conv2.pesos = pesos['conv2_pesos']
        self.conv2.bias = pesos['conv2_bias']
        self.fc1.pesos = pesos['fc1_pesos']
        self.fc1.bias = pesos['fc1_bias']
        self.fc2.pesos = pesos['fc2_pesos']
        self.fc2.bias = pesos['fc2_bias']
        
        print(f"✓ Pesos cargados desde {ruta}")