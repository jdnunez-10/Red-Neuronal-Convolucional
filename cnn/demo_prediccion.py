import os
import numpy as np
from PIL import Image
from src.modelo import CNNMinima

TAMAÑO_IMG = (128, 128)   # Debe coincidir con tu entrenamiento

# -------------------------------------------------------------
# Cargar una imagen individual
# -------------------------------------------------------------
def cargar_imagen(ruta, tamaño):
    img = Image.open(ruta).convert("L")
    img = img.resize(tamaño)
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.reshape(1, 1, tamaño[0], tamaño[1])
    return img


# -------------------------------------------------------------
# DEMO DE PREDICCIÓN
# -------------------------------------------------------------
def demo_prediccion():

    print("\n======================================================================")
    print(" DEMO: CLASIFICADOR CNN (5 CLASES + NINGUNA)")
    print("======================================================================\n")

    # Inicializar modelo
    modelo = CNNMinima(tamaño_entrada=(1, 128, 128))
    modelo.cargar_pesos("pesos_finales.pkl")
    print("✓ Modelo cargado exitosamente\n")

    # Pedir ruta de imagen
    ruta = input("Ruta de la imagen a predecir: ").strip()

    if os.path.isdir(ruta):
        raise ValueError("ERROR: La ruta es una CARPETA. Debe seleccionar un archivo de imagen.")

    if not os.path.exists(ruta):
        raise FileNotFoundError("No existe la imagen indicada.")

    print("\nCargando imagen:", ruta)
    img = cargar_imagen(ruta, TAMAÑO_IMG)

    # -------------------------------------------------------------
    #             REALIZAR LA PREDICCIÓN
    # -------------------------------------------------------------

    # === 3. Forward ===
    logits = modelo.forward(img)

    # convertir logits a probabilidades
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)

    indice_pred = int(np.argmax(probs))
    confianza = float(np.max(probs))

    etiquetas = ["Bike", "Car", "Gato", "Human","Perro"]  
    pred, probs = modelo.predecir_con_umbral(img)

    print("\n================ RESULTADO ================")
    print("Clase predicha:", pred[0])
    print("Probabilidades:", probs[0])
    print(f"Confianza: {confianza*100:.2f}%")
    print("===========================================\n")


if __name__ == "__main__":
    demo_prediccion()
