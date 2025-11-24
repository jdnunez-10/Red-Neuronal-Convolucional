import numpy as np
import pickle
from src.capas.capaConvolucional import CapaConvolucional
from src.capas.capaPooling import CapaMaxPooling2x2
from src.capas.capaAplanada import CapaAplanada
from src.capas.capaTotalmenteConectada import CapaTotalmenteConectada
from src.activaciones import ReLU, Softmax
from src.funcionesPerdida import funcion_perdida_crossentropy, derivada_crossentropy


class CNNMinima:
    """
    Red Neuronal Convolucional programada desde cero.
    Arquitectura optimizada para imágenes 224x224:

        Conv1 (4 filtros) → ReLU → Pool1
        Conv2 (8 filtros) → ReLU → Pool2
        Flatten → FC1 → ReLU → FC2 → Softmax
    """

    def __init__(self, tamaño_entrada=(1, 128, 128), num_clases=5, lr=5e-3, umbral=0.6):
        C, H, W = tamaño_entrada
        self.lr = lr
        self.umbral = umbral
        self.num_clases = num_clases

        print("="*70)
        print("  INICIALIZANDO CNN PARA 128x128")
        print("="*70)
        print(f"Entrada: {tamaño_entrada}")
        print(f"Clases: {num_clases}\n")

        # --- BLOQUE 1 ---
        print(" Bloque 1: 8 filtros")
        self.conv1 = CapaConvolucional(tamaño_entrada[0], 8, tam_kernel=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = CapaMaxPooling2x2()

        # --- BLOQUE 2 ---
        print(" Bloque 2: 16 filtros")
        self.conv2 = CapaConvolucional(8, 16, tam_kernel=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = CapaMaxPooling2x2()

        # --- BLOQUE 3 ---
        print(" Bloque 3: 32 filtros")
        self.conv3 = CapaConvolucional(16, 32, tam_kernel=3, padding=1)
        self.relu3 = ReLU()
        self.pool3 = CapaMaxPooling2x2()

        # --- BLOQUE 4 ---
        print(" Bloque 4: 64 filtros")
        self.conv4 = CapaConvolucional(32, 64, tam_kernel=3, padding=1)
        self.relu4 = ReLU()
        self.pool4 = CapaMaxPooling2x2()

        # --- APLANADO ---
        self.flatten = CapaAplanada()

        # --- CALCULAR TAMAÑO AUTOMÁTICAMENTE ---
        print("\n Calculando dimensiones...")
        dummy = np.zeros((1, *tamaño_entrada))
        
        x = self.conv1.forward(dummy)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        print(f"   Después Pool1: {x.shape}")

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        print(f"   Después Pool2: {x.shape}")

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)
        print(f"   Después Pool3: {x.shape}")

        x = self.conv4.forward(x)
        x = self.relu4.forward(x)
        x = self.pool4.forward(x)
        print(f"   Después Pool4: {x.shape}")

        x = self.flatten.forward(x)
        tamaño_flat = x.shape[1]
        print(f"\n   ✓ Vector aplanado: {tamaño_flat:,}\n")

        # --- Fully Connected ---
        print(f" FC1: {tamaño_flat:,} → 256")
        self.fc1 = CapaTotalmenteConectada(tamaño_flat, 256)
        self.relu_fc = ReLU()

        print(f" FC2: 256 → {num_clases}")
        self.fc2 = CapaTotalmenteConectada(256, num_clases)
        self.softmax = Softmax()

        # Registrar capas en orden
        self.capas = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.conv3, self.relu3, self.pool3,
            self.conv4, self.relu4, self.pool4,
            self.flatten,
            self.fc1, self.relu_fc,
            self.fc2
        ]

        # Calcular total de parámetros
        total_params = 0
        for capa in self.capas:
            if hasattr(capa, 'pesos'):
                total_params += np.prod(capa.pesos.shape)
                if hasattr(capa, 'bias'):
                    total_params += np.prod(capa.bias.shape)

        print("\n" + "="*70)
        print("   CNN INICIALIZADA")
        print("="*70)
        print(f"Total parámetros: {total_params:,}")
        print(f"Capas: {len(self.capas)}")
        print("="*70 + "\n")

    # ------------------------------
    # FORWARD
    # ------------------------------
    def forward(self, x):
        for capa in self.capas:
            x = capa.forward(x)
        return x
# ------------------------------
    # BACKWARD
    # ------------------------------
    def backward(self, grad, lr):
        """Backpropagation con learning rate"""
        for capa in reversed(self.capas):
            if hasattr(capa, 'backward'):
                # Algunas capas necesitan lr, otras no
                if isinstance(capa, (CapaConvolucional, CapaTotalmenteConectada)):
                    grad = capa.backward(grad, lr)
                else:
                    grad = capa.backward(grad)
        return grad

    # ------------------------------
    # ACTUALIZACIÓN DE PESOS
    # ------------------------------
    def actualizar(self, lr=5e-3):
        """Actualiza pesos de capas que lo necesitan"""
        for capa in self.capas:
            if hasattr(capa, "actualizar"):
                capa.actualizar(lr)

    # ------------------------------
    # ENTRENAMIENTO POR LOTE
    # ------------------------------
    def paso_entrenamiento(self, x_batch, y_batch):
        """Un paso completo de entrenamiento"""
        # Forward pass
        output = self.forward(x_batch)
        
        # Calcular pérdida
        perdida = funcion_perdida_crossentropy(output, y_batch)
        
        # Backward pass
        grad = derivada_crossentropy(output, y_batch)
        self.backward(grad)
        
        # Actualizar pesos
        self.actualizar()
        
        return perdida, output

    # ------------------------------
    # PREDICCIÓN
    # ------------------------------
    def predecir(self, x):
        """Predicción en modo evaluación"""
        scores = self.forward(x, entrenamiento=False)
        return np.argmax(scores, axis=1)

    def predecir_proba(self, x):
        """Retorna probabilidades"""
        return self.forward(x, entrenamiento=False)
    # ------------------------------
    # GUARDAR Y CARGAR PESOS
    # ------------------------------
    def guardar_pesos(self, ruta):
        import pickle
        pesos = {}
        for i, capa in enumerate(self.capas):
            if hasattr(capa, 'pesos'):
                pesos[f'capa_{i}_pesos'] = capa.pesos
            if hasattr(capa, 'bias'):
                pesos[f'capa_{i}_bias'] = capa.bias
        
        with open(ruta, 'wb') as f:
            pickle.dump(pesos, f)
        print(f"✓ Pesos guardados en {ruta}")

    def cargar_pesos(self, ruta):
        import pickle
        with open(ruta, 'rb') as f:
            pesos = pickle.load(f)
        
        for i, capa in enumerate(self.capas):
            if hasattr(capa, 'pesos') and f'capa_{i}_pesos' in pesos:
                capa.pesos = pesos[f'capa_{i}_pesos']
            if hasattr(capa, 'bias') and f'capa_{i}_bias' in pesos:
                capa.bias = pesos[f'capa_{i}_bias']
        
        print(f"✓ Pesos cargados desde {ruta}")