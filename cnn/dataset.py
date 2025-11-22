import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

class CargadorDataset:
    """
    Cargador de dataset de imágenes organizado por carpetas.
    Estructura esperada:

        dataset/
            gato/
            perro/
            tigre/
            caballo/
            ave/

    Cada carpeta representa una clase.
    """

    def __init__(self, ruta_dataset, tamaño=(700, 600), usar_grises=True):
        self.ruta_dataset = ruta_dataset
        self.tamaño = tamaño  # (alto, ancho)
        self.usar_grises = usar_grises
        self.mapa_clases = {}  # {"gato":0, "perro":1, ...}

    # -----------------------------------------------------
    #                 Cargar imágenes
    # -----------------------------------------------------
    def cargar_imagen(self, ruta_imagen):
        """
        Lee, convierte y normaliza una imagen.
        Devuelve un tensor con forma (1, alto, ancho)
        """
        img = Image.open(ruta_imagen)

        if self.usar_grises:
            img = img.convert("L")  # convertir a escala de grises

        img = img.resize(self.tamaño[::-1])  # invertir (alto, ancho) → (ancho, alto)

        img_array = np.array(img, dtype=np.float32) / 255.0  # normalizar
        img_array = img_array.reshape(1, self.tamaño[0], self.tamaño[1])

        return img_array

    # -----------------------------------------------------
    #                Cargar dataset completo
    # -----------------------------------------------------
    def cargar_dataset(self):
        """
        Recorre todas las carpetas y carga las imágenes.
        Asigna etiquetas numéricas automáticamente.
        """

        imagenes = []
        etiquetas = []

        clases = sorted(os.listdir(self.ruta_dataset))
        self.mapa_clases = {clase: i for i, clase in enumerate(clases)}

        print("Clases detectadas:", self.mapa_clases)

        for clase in clases:
            carpeta_clase = os.path.join(self.ruta_dataset, clase)

            for archivo in os.listdir(carpeta_clase):
                ruta_archivo = os.path.join(carpeta_clase, archivo)

                if not archivo.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    continue  # ignorar archivos no imagen

                img = self.cargar_imagen(ruta_archivo)
                imagenes.append(img)
                etiquetas.append(self.mapa_clases[clase])

        # Convertir a arrays
        X = np.array(imagenes, dtype=np.float32)
        y = np.array(etiquetas, dtype=np.int32)

        # Mezclar
        X, y = shuffle(X, y, random_state=42)

        return X, y, self.mapa_clases

    # -----------------------------------------------------
    #             División de entrenamiento / prueba
    # -----------------------------------------------------
    def dividir_train_test(self, X, y, porcentaje_test=0.2):
        total = len(X)
        n_test = int(total * porcentaje_test)

        X_train = X[:-n_test]
        y_train = y[:-n_test]
        X_test = X[-n_test:]
        y_test = y[-n_test:]

        return X_train, y_train, X_test, y_test
