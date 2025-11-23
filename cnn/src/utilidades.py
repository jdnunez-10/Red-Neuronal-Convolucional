import numpy as np
import os
from PIL import Image

def cargar_dataset(ruta_data, tam_imagen=(700,600), max_por_clase=100):
    X = []
    y = []
    clases = sorted(os.listdir(ruta_data))
    for idx, clase in enumerate(clases):
        ruta_clase = os.path.join(ruta_data, clase)
        imagenes = os.listdir(ruta_clase)[:max_por_clase]
        for img_nombre in imagenes:
            img = Image.open(os.path.join(ruta_clase, img_nombre)).convert('L')  # escala de grises
            img = img.resize(tam_imagen)
            X.append(np.array(img, dtype=np.float32)/255.0)
            y.append(idx)
    X = np.array(X)
    X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
    y = np.array(y)
    return X, y, clases

def one_hot(y, num_clases):
    one_hot = np.zeros((y.shape[0], num_clases))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def generar_mini_batches(X, y, batch_size=16):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]
