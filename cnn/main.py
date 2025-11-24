import numpy as np
from src.modelo import CNNMinima
from entrenamiento import EntrenadorCNN
from dataset import CargadorDataset
from src.funcionesPerdida import funcion_perdida_crossentropy, derivada_crossentropy

if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. CARGAR DATASET
    # ------------------------------------------------------------
    ruta_dataset = "dataset/"
    print("="*70)
    print("  CARGANDO DATASET")
    print("="*70)

    cargador = CargadorDataset(ruta_dataset, tamaño=(128, 128))  # tamaño oficial
    X, y, mapa_clases = cargador.cargar_dataset(max_por_clase=40)

    print(f"\nMapa de clases detectado: {mapa_clases}")
    print(f"Total de imágenes cargadas: {len(X)}")
    print(f"Forma de las imágenes: {X.shape}")

    X_train, y_train, X_test, y_test = cargador.dividir_train_test(X, y, porcentaje_test=0.2)

    print(f"\n{'─'*70}")
    print(f"Imágenes de entrenamiento: {X_train.shape[0]}")
    print(f"Imágenes de prueba: {X_test.shape[0]}")
    print(f"{'─'*70}\n")

    # ------------------------------------------------------------
    # 2. CREAR MODELO CNN
    # ------------------------------------------------------------
    modelo = CNNMinima(
        tamaño_entrada=(1, 128, 128),
        num_clases=len(mapa_clases),
        lr=5e-4,
        umbral=0.6
    )

    print("="*70)
    print("  RED NEURONAL CONVOLUCIONAL LISTA")
    print("="*70 + "\n")

    # ------------------------------------------------------------
    # 3. CONFIGURAR ENTRENADOR
    # ------------------------------------------------------------
    entrenador = EntrenadorCNN(
        modelo=modelo,
        lr=5e-4,
        funcion_perdida=funcion_perdida_crossentropy,
        derivada_funcion_perdida=derivada_crossentropy
    )

    # ------------------------------------------------------------
    # 4. ENTRENAMIENTO
    # ------------------------------------------------------------
    entrenador.entrenar(
        X_train,
        y_train,
        epochs=15,
        lote_size=32,
        validacion=(X_test, y_test)
    )

    # ------------------------------------------------------------
    # 5. EVALUACIÓN FINAL
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("  EVALUACIÓN FINAL EN TEST")
    print("="*70)

    accuracy_test = entrenador.evaluar(X_test, y_test, lote_size=32)
    print(f"\nAccuracy en test: {accuracy_test*100:.2f}%")

    # Ejemplos
    logits = modelo.forward(X_test[:10])
    predicciones = np.argmax(logits, axis=1)
    mapa_inverso = {v: k for k, v in mapa_clases.items()}

    for i in range(len(predicciones)):
        real = mapa_inverso[y_test[i]]
        pred = mapa_inverso[predicciones[i]]
        correcto = "✓" if y_test[i] == predicciones[i] else "✗"
        print(f"{correcto} Ejemplo {i+1}: Real='{real}' | Predicho='{pred}'")

    # ------------------------------------------------------------
    # 6. GUARDAR PESOS
    # ------------------------------------------------------------
    modelo.guardar_pesos("pesos_finales.pkl")

    # ------------------------------------------------------------
    # 7. MOSTRAR HISTORIAL
    # ------------------------------------------------------------
    entrenador.mostrar_historial()