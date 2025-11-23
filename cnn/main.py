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
    
    # Configuración balanceada
    cargador = CargadorDataset(ruta_dataset, tamaño=(64, 64))
    X, y, mapa_clases = cargador.cargar_dataset(max_por_clase=300)  # 500 por clase = 1000 total

    print(f"\nMapa de clases detectado: {mapa_clases}")
    print(f"Total de imágenes cargadas: {len(X)}")
    print(f"Forma de las imágenes: {X.shape}")

    # Dividir en train y test
    X_train, y_train, X_test, y_test = cargador.dividir_train_test(X, y, porcentaje_test=0.2)

    print(f"\n{'─'*70}")
    print(f"Imágenes de entrenamiento: {X_train.shape[0]}")
    print(f"Imágenes de prueba: {X_test.shape[0]}")
    print(f"{'─'*70}\n")


    # ------------------------------------------------------------
    # 2. CREAR MODELO CNN
    # ------------------------------------------------------------
    print("="*70)
    print("  CONSTRUYENDO RED NEURONAL CONVOLUCIONAL")
    print("="*70)
    
    modelo = CNNMinima(
        tamaño_entrada=(1, 64, 64),
        num_clases=len(mapa_clases),
        lr=0.01,  # Learning rate MÁS ALTO
        umbral=0.6
    )
    
    print(f"Número de clases: {len(mapa_clases)}")
    print(f"Arquitectura: Conv→ReLU→Pool → Conv→ReLU→Pool → Flatten → FC→ReLU → FC")
    print("="*70 + "\n")


    # ------------------------------------------------------------
    # 3. CONFIGURAR ENTRENADOR
    # ------------------------------------------------------------
    entrenador = EntrenadorCNN(
        modelo=modelo,
        lr=0.1,  # Learning rate MÁS ALTO (10x mayor)
        funcion_perdida=funcion_perdida_crossentropy,
        derivada_funcion_perdida=derivada_crossentropy
    )


    # ------------------------------------------------------------
    # 4. ENTRENAMIENTO
    # ------------------------------------------------------------

    
    entrenador.entrenar(
        X_train,
        y_train,
        epochs=10,  # MÁS ÉPOCAS
        lote_size=64,
        validacion=(X_test, y_test)
    )


    # ------------------------------------------------------------
    # 5. EVALUACIÓN FINAL
    # ------------------------------------------------------------
    print("\n" + "="*70)
    print("  EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA")
    print("="*70)
    
    accuracy_test = entrenador.evaluar(X_test, y_test, lote_size=64)
    print(f"\nAccuracy en test: {accuracy_test*100:.2f}%")
    
    # Mostrar algunas predicciones
    print("\n" + "─"*70)
    print("  EJEMPLOS DE PREDICCIONES")
    print("─"*70)
    
    num_ejemplos = min(10, len(X_test))  # Mostrar 10 ejemplos
    logits = modelo.forward(X_test[:num_ejemplos])
    predicciones = np.argmax(logits, axis=1)
    
    # Invertir el mapa de clases para mostrar nombres
    mapa_inverso = {v: k for k, v in mapa_clases.items()}
    
    aciertos = 0
    for i in range(num_ejemplos):
        clase_real = mapa_inverso[y_test[i]]
        clase_pred = mapa_inverso[predicciones[i]]
        correcto = "Correcto" if y_test[i] == predicciones[i] else "Incorrecto"
        if y_test[i] == predicciones[i]:
            aciertos += 1
        print(f"{correcto} Ejemplo {i+1}: Real='{clase_real}' | Predicho='{clase_pred}'")
    
    print(f"\nAciertos en ejemplos: {aciertos}/{num_ejemplos}")
    print("─"*70 + "\n")


    # ------------------------------------------------------------
    # 6. GUARDAR PESOS
    # ------------------------------------------------------------
    print("="*70)
    print("  GUARDANDO MODELO")
    print("="*70)
    
    modelo.guardar_pesos("pesos_finales.pkl")
    print("\nModelo guardado exitosamente en 'pesos_finales.pkl'")
    print("="*70 + "\n")
    
    
    # ------------------------------------------------------------
    # 7. MOSTRAR HISTORIAL
    # ------------------------------------------------------------
    entrenador.mostrar_historial()


    print("\n" + "="*70)
    print("  ¡ENTRENAMIENTO COMPLETADO!")
    print("="*70)
    
    # Análisis de resultados
    accuracy_final = entrenador.historial['accuracy'][-1]
    
    if accuracy_final > 0.70:
        print("\n ¡EXCELENTE! El modelo aprendió bien (>70% accuracy)")
    elif accuracy_final > 0.60:
        print("\n✓ BIEN. El modelo está aprendiendo (60-70% accuracy)")
        print(" Tip: Aumenta epochs a 20 o más imágenes para mejorar")
    elif accuracy_final > 0.50:
        print("\n REGULAR. El modelo aprende poco (50-60% accuracy)")
        print(" Tips:")
        print("   - Aumenta learning rate a 0.05")
        print("   - Usa más imágenes (500+ por clase)")
        print("   - Entrena más épocas (20+)")
    else:
        print("\n El modelo NO está aprendiendo (<50% accuracy)")
        print("💡 Posibles problemas:")
        print("   - Learning rate muy bajo → Prueba 0.05 o 0.1")
        print("   - Muy pocas imágenes → Usa al menos 500 por clase")
        print("   - Imágenes muy pequeñas → Prueba 64x64 o más")
    
    print("="*70 + "\n")