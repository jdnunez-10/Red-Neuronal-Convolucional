"""
Script de DEMOSTRACIÓN para tu defensa.
Carga un modelo ya entrenado y hace predicciones instantáneas.
"""

import numpy as np
from PIL import Image
from src.modelo import CNNMinima
import os

def cargar_imagen_para_prediccion(ruta_imagen, tamaño=(64, 64)):
    """
    Carga y preprocesa una imagen para predicción.
    """
    # Abrir imagen
    img = Image.open(ruta_imagen)
    
    # Convertir a escala de grises
    img = img.convert('L')
    
    # Redimensionar
    img = img.resize(tamaño[::-1])  # PIL usa (width, height)
    
    # Convertir a array y normalizar
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Dar forma (1, 1, H, W) para batch de 1 imagen
    img_array = img_array.reshape(1, 1, tamaño[0], tamaño[1])
    
    return img_array


def demo_predicciones_rapidas():
    """
    Demostración rápida de predicciones con modelo pre-entrenado.
    """
    print("\n" + "="*70)
    print(" DEMO: CLASIFICADOR DE PERROS Y GATOS")
    print("="*70 + "\n")
    
    # Configuración
    TAMAÑO_IMG = (128, 128)
    ARCHIVO_PESOS = "pesos_finales.pkl"
    # Ajusta este mapeo según el orden que detectó tu entrenamiento
    MAPA_CLASES = {0: 'bike', 1: 'car', 2: 'gato', 3: 'human', 4: 'perro'}  # Orden alfabético
    
    # Verificar que existen los pesos
    if not os.path.exists(ARCHIVO_PESOS):
        print(f" ERROR: No se encontró '{ARCHIVO_PESOS}'")
        print("\n Primero debes entrenar el modelo:")
        print("   python main.py")
        print("\nEsto generará el archivo de pesos.\n")
        return
    
    # ============================================================
    # 1. CARGAR MODELO
    # ============================================================
    print(" Cargando modelo pre-entrenado...")
    
    modelo = CNNMinima(
        tamaño_entrada=(1, TAMAÑO_IMG[0], TAMAÑO_IMG[1]),
        num_clases=len(MAPA_CLASES),
        lr=0.05,
        umbral=0.6
    )
    
    modelo.cargar_pesos(ARCHIVO_PESOS)
    
    print(" Modelo cargado exitosamente!\n")
    
    # ============================================================
    # 2. MODO INTERACTIVO
    # ============================================================
    print("="*70)
    print("  MODO DE PREDICCIÓN")
    print("="*70 + "\n")
    
    print("Opciones:")
    print("  1. Predecir una imagen específica")
    print("  2. Predecir imágenes del dataset de prueba")
    print("  3. Predecir todas las imágenes de una carpeta")
    print()
    
    opcion = input("Selecciona una opción (1-3): ").strip()
    
    # ============================================================
    # OPCIÓN 1: Imagen específica
    # ============================================================
    if opcion == "1":
        print("\n" + "─"*70)
        ruta = input("Ruta de la imagen: ").strip().strip('"').strip("'")
        
        if not os.path.exists(ruta):
            print(f" No se encuentra la imagen: {ruta}")
            return
        
        print("\n🔍 Analizando imagen...\n")
        
        # Cargar y predecir
        img = cargar_imagen_para_prediccion(ruta, TAMAÑO_IMG)
        logits = modelo.forward(img)
        
        # Aplicar softmax para obtener probabilidades
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Predicción
        clase_pred = np.argmax(probs[0])
        confianza = probs[0][clase_pred]
        
        print("="*70)
        print("  RESULTADO DE LA PREDICCIÓN")
        print("="*70)
        print(f"\n Predicción: {MAPA_CLASES[clase_pred].upper()}")
        print(f" Confianza: {confianza*100:.2f}%")
        print(f"\n Probabilidades:")
        for idx, nombre in MAPA_CLASES.items():
            print(f"   {nombre}: {probs[0][idx]*100:.2f}%")
        print("\n" + "="*70 + "\n")
    
    # ============================================================
    # OPCIÓN 2: Dataset de prueba
    # ============================================================
    elif opcion == "2":
        from dataset import CargadorDataset
        
        print("\n Cargando dataset de prueba...")
        
        cargador = CargadorDataset("dataset/", tamaño=TAMAÑO_IMG)
        X, y, mapa_clases = cargador.cargar_dataset(max_por_clase=50)
        _, _, X_test, y_test = cargador.dividir_train_test(X, y, porcentaje_test=0.3)
        
        print(f"✓ {len(X_test)} imágenes de prueba cargadas\n")
        
        # Predecir en lote
        print(" Realizando predicciones...\n")
        logits = modelo.forward(X_test)
        predicciones = np.argmax(logits, axis=1)
        
        # Calcular accuracy
        accuracy = np.mean(predicciones == y_test)
        
        print("="*70)
        print("  RESULTADOS EN DATASET DE PRUEBA")
        print("="*70)
        print(f"\n Accuracy: {accuracy*100:.2f}%")
        print(f"✓ Aciertos: {np.sum(predicciones == y_test)}/{len(y_test)}")
        print(f"✗ Errores: {np.sum(predicciones != y_test)}/{len(y_test)}")
        
        # Mostrar 10 ejemplos
        print(f"\n{'─'*70}")
        print("  EJEMPLOS DE PREDICCIONES")
        print("─"*70 + "\n")
        
        mapa_inverso = {v: k for k, v in mapa_clases.items()}
        
        for i in range(min(10, len(X_test))):
            real = mapa_inverso[y_test[i]]
            pred = mapa_inverso[predicciones[i]]
            correcto = "✓" if y_test[i] == predicciones[i] else "✗"
            print(f"{correcto} Imagen {i+1}: Real='{real}' | Predicho='{pred}'")
        
        print("\n" + "="*70 + "\n")
    
    # ============================================================
    # OPCIÓN 3: Carpeta completa
    # ============================================================
    elif opcion == "3":
        print("\n" + "─"*70)
        carpeta = input("Ruta de la carpeta: ").strip().strip('"').strip("'")
        
        if not os.path.exists(carpeta):
            print(f" No se encuentra la carpeta: {carpeta}")
            return
        
        # Buscar imágenes
        imagenes = [f for f in os.listdir(carpeta) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not imagenes:
            print(f" No se encontraron imágenes en: {carpeta}")
            return
        
        print(f"\n Analizando {len(imagenes)} imágenes...\n")
        
        resultados = {'gato': 0, 'perro': 0}
        
        print("─"*70)
        for i, img_nombre in enumerate(imagenes[:20], 1):  # Máximo 20 para demo
            ruta_img = os.path.join(carpeta, img_nombre)
            
            try:
                img = cargar_imagen_para_prediccion(ruta_img, TAMAÑO_IMG)
                logits = modelo.forward(img)
                clase_pred = np.argmax(logits)
                
                nombre_clase = MAPA_CLASES[clase_pred]
                resultados[nombre_clase] += 1
                
                print(f"{i:2d}. {img_nombre:<30} → {nombre_clase.upper()}")
                
            except Exception as e:
                print(f"{i:2d}. {img_nombre:<30} → ERROR: {e}")
        
        print("─"*70)
        print(f"\n📊 RESUMEN:")
        print(f"   Gatos detectados: {resultados['gato']}")
        print(f"   Perros detectados: {resultados['perro']}")
        print("="*70 + "\n")
    
    else:
        print("\n Opción inválida\n")


def demo_automatica():
    """
    Demo completamente automática - ideal para defensa.
    Usa imágenes de ejemplo del dataset.
    """
    print("\n" + "="*70)
    print(" DEMO AUTOMÁTICA - CLASIFICADOR DOGS VS CATS")
    print("="*70 + "\n")
    
    from dataset import CargadorDataset
    
    # Cargar modelo
    print("1 Cargando modelo entrenado...")
    modelo = CNNMinima(
        tamaño_entrada=(1, 64, 64),
        num_clases=2,
        lr=0.01,
        umbral=0.6
    )
    modelo.cargar_pesos("pesos_finales.pkl")
    print("   ✓ Modelo cargado\n")
    
    # Cargar imágenes de prueba
    print("2 Cargando imágenes de prueba...")
    cargador = CargadorDataset("dataset/", tamaño=(128, 128))
    X, y, mapa_clases = cargador.cargar_dataset(max_por_clase=50)
    _, _, X_test, y_test = cargador.dividir_train_test(X, y, porcentaje_test=0.5)
    print(f"    {len(X_test)} imágenes de prueba cargadas\n")
    
    # Realizar predicciones
    print("3 Realizando predicciones...")
    logits = modelo.forward(X_test)
    predicciones = np.argmax(logits, axis=1)
    accuracy = np.mean(predicciones == y_test)
    print(f"    Predicciones completadas\n")
    
    # Mostrar resultados
    print("="*70)
    print("   RESULTADOS")
    print("="*70)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Aciertos: {np.sum(predicciones == y_test)}/{len(y_test)}\n")
    
    # Ejemplos
    print("─"*70)
    print("  EJEMPLOS DE CLASIFICACIÓN")
    print("─"*70 + "\n")
    
    mapa_inverso = {v: k for k, v in mapa_clases.items()}
    
    for i in range(min(10, len(X_test))):
        real = mapa_inverso[y_test[i]]
        pred = mapa_inverso[predicciones[i]]
        correcto = "✓" if y_test[i] == predicciones[i] else "✗"
        print(f"{correcto} Imagen {i+1}: Real='{real.upper()}' → Predicho='{pred.upper()}'")
    
    print("\n" + "="*70)
    print("   DEMOSTRACIÓN COMPLETADA")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n MODO DE DEMOSTRACIÓN\n")
    print("Selecciona el modo:")
    print("  1. Demo interactiva (tú eliges qué predecir)")
    print("  2. Demo automática (para defensa rápida)")
    print()
    
    modo = input("Modo (1-2): ").strip()
    
    if modo == "1":
        demo_predicciones_rapidas()
    elif modo == "2":
        demo_automatica()
    else:
        print("\n Opción inválida\n")