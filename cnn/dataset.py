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

    def __init__(self, ruta_dataset, tamaño=(128, 128), usar_grises=True):
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
    def cargar_dataset(self, max_por_clase=500):
        """
        Recorre todas las carpetas y carga las imágenes.
        Asigna etiquetas numéricas automáticamente.
        
        Parámetros:
        -----------
        max_por_clase : int
            Número máximo de imágenes a cargar por clase.
            Usa None para cargar todas (puede ser muy lento).
        """

        imagenes = []
        etiquetas = []

        # Obtener solo las CARPETAS (no archivos) y ordenarlas
        # Ignorar carpetas especiales
        contenido = os.listdir(self.ruta_dataset)
        clases = sorted([
            item for item in contenido 
            if os.path.isdir(os.path.join(self.ruta_dataset, item))
            and item not in ['__pycache__', '.git', '.ipynb_checkpoints']
        ])

        # Crear mapeo de clases a índices
        self.mapa_clases = {clase: i for i, clase in enumerate(clases)}

        print("Clases detectadas:", self.mapa_clases)
        if max_por_clase:
            print(f"Límite por clase: {max_por_clase} imágenes")
        print()

        # Procesar cada clase
        for clase in clases:
            carpeta_clase = os.path.join(self.ruta_dataset, clase)
            
            print(f"Cargando clase '{clase}'...", end='', flush=True)
            
            contador = 0
            
            # Listar archivos en la carpeta de la clase
            archivos = os.listdir(carpeta_clase)
            
            for archivo in archivos:
                # Verificar límite por clase
                if max_por_clase and contador >= max_por_clase:
                    break
                ruta_archivo = os.path.join(carpeta_clase, archivo)

                # Verificar que sea un archivo (no una carpeta)
                if not os.path.isfile(ruta_archivo):
                    continue

                # Verificar extensión de imagen
                if not archivo.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                    continue

                try:
                    # Cargar imagen
                    img = self.cargar_imagen(ruta_archivo)
                    imagenes.append(img)
                    etiquetas.append(self.mapa_clases[clase])
                    contador += 1
                    
                    # Mostrar progreso cada 100 imágenes
                    if contador % 100 == 0:
                        print(f"\r  Cargando clase '{clase}'... {contador} imágenes", end='', flush=True)
                    
                except Exception as e:
                    print(f"\r  ⚠️  Error al cargar {archivo}: {e}")
                    continue
            
            print(f"\r  ✓ {clase}: {contador} imágenes cargadas" + " "*20)

        # Verificar que se hayan cargado imágenes
        if len(imagenes) == 0:
            raise ValueError(
                "No se encontraron imágenes válidas en el dataset. "
                "Verifica que:\n"
                "  1. Las carpetas contengan imágenes (.jpg, .png, etc.)\n"
                "  2. La ruta del dataset sea correcta\n"
                f"  3. Ruta actual: {self.ruta_dataset}"
            )

        # Convertir a arrays
        X = np.array(imagenes, dtype=np.float32)
        y = np.array(etiquetas, dtype=np.int32)

        print(f"Total de imágenes cargadas: {len(X)}")

        # Mezclar
        X, y = shuffle(X, y, random_state=42)

        return X, y, self.mapa_clases

    # -----------------------------------------------------
    #             División de entrenamiento / prueba
    # -----------------------------------------------------
    def dividir_train_test(self, X, y, porcentaje_test=0.8):
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Imágenes
        y : numpy.ndarray
            Etiquetas
        porcentaje_test : float
            Porcentaje de datos para prueba (0.0 a 1.0)
        
        Retorna:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
        total = len(X)
        n_test = int(total * porcentaje_test)

        # Asegurar que haya al menos 1 muestra en test
        if n_test < 1:
            n_test = 1

        X_train = X[:-n_test]
        y_train = y[:-n_test]
        X_test = X[-n_test:]
        y_test = y[-n_test:]

        return X_train, y_train, X_test, y_test