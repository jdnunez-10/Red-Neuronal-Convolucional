import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

class CargadorDataset:
    """
    Cargador de dataset de imágenes organizado por carpetas.
    Cada carpeta representa una clase.
    """

    def __init__(self, ruta_dataset, tamaño=(128, 128), usar_grises=True):
        self.ruta_dataset = ruta_dataset
        self.tamaño = tamaño  # (alto, ancho)
        self.usar_grises = usar_grises
        self.mapa_clases = {}  # {"gato":0, "perro":1, ...}

    def cargar_imagen(self, ruta_imagen):
        """Lee, convierte y normaliza una imagen."""
        img = Image.open(ruta_imagen)

        if self.usar_grises:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = img.resize(self.tamaño[::-1])  # invertir (alto, ancho) → (ancho, alto)

        img_array = np.array(img, dtype=np.float32) / 255.0
        if self.usar_grises:
            img_array = img_array.reshape(1, self.tamaño[0], self.tamaño[1])
        else:
            img_array = img_array.transpose(2, 0, 1)  # (C,H,W)

        return img_array

    def cargar_dataset(self, max_por_clase=40):
        """Carga todas las imágenes y asigna etiquetas numéricas automáticamente."""
        imagenes, etiquetas = [], []

        contenido = os.listdir(self.ruta_dataset)
        clases = sorted([
            item for item in contenido 
            if os.path.isdir(os.path.join(self.ruta_dataset, item))
            and item not in ['__pycache__', '.git', '.ipynb_checkpoints']
        ])

        self.mapa_clases = {clase: i for i, clase in enumerate(clases)}

        print("Clases detectadas:", self.mapa_clases)

        for clase in clases:
            carpeta_clase = os.path.join(self.ruta_dataset, clase)
            archivos = os.listdir(carpeta_clase)
            contador = 0

            for archivo in archivos:
                if max_por_clase and contador >= max_por_clase:
                    break
                ruta_archivo = os.path.join(carpeta_clase, archivo)
                if not os.path.isfile(ruta_archivo):
                    continue
                if not archivo.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    continue

                try:
                    img = self.cargar_imagen(ruta_archivo)
                    imagenes.append(img)
                    etiquetas.append(self.mapa_clases[clase])
                    contador += 1
                except Exception as e:
                    print(f"Error al cargar {archivo}: {e}")
                    continue

            print(f"   {clase}: {contador} imágenes cargadas")

        if len(imagenes) == 0:
            raise ValueError("No se encontraron imágenes válidas en el dataset.")

        X = np.array(imagenes, dtype=np.float32)
        y = np.array(etiquetas, dtype=np.int32)

        print(f"Total de imágenes cargadas: {len(X)}")

        X, y = shuffle(X, y, random_state=42)
        return X, y, self.mapa_clases

    def dividir_train_test(self, X, y, porcentaje_test=0.2):
        """Divide el dataset en conjuntos de entrenamiento y prueba."""
        total = len(X)
        n_test = int(total * porcentaje_test)
        if n_test < 1:
            n_test = 1

        X_train, y_train = X[:-n_test], y[:-n_test]
        X_test, y_test = X[-n_test:], y[-n_test:]
        return X_train, y_train, X_test, y_test