"""
Script para reorganizar el dataset Dogs vs Cats de Kaggle
en la estructura que necesita tu CNN.
"""

import os
import shutil
from pathlib import Path

def organizar_dogs_vs_cats(ruta_original, ruta_destino="dataset", limite_por_clase=None):
    """
    Reorganiza Dogs vs Cats de Kaggle en carpetas separadas.
    
    Parámetros:
    -----------
    ruta_original : str
        Ruta donde está el dataset descargado de Kaggle
        Ejemplo: "dogs-vs-cats/train"
    ruta_destino : str
        Carpeta donde se creará la estructura organizada
    limite_por_clase : int, opcional
        Limitar cantidad de imágenes por clase (útil para pruebas rápidas)
    """
    
    print("\n" + "="*70)
    print("  ORGANIZANDO DATASET DOGS VS CATS")
    print("="*70 + "\n")
    
    # Verificar que existe la ruta original
    if not os.path.exists(ruta_original):
        print(f"❌ Error: No se encuentra la ruta '{ruta_original}'")
        print("\nAsegúrate de:")
        print("  1. Haber descargado el dataset de Kaggle")
        print("  2. Haber descomprimido el archivo .zip")
        print("  3. Especificar la ruta correcta")
        return False
    
    # Crear estructura de carpetas
    carpeta_perros = os.path.join(ruta_destino, "perro")
    carpeta_gatos = os.path.join(ruta_destino, "gato")
    
    os.makedirs(carpeta_perros, exist_ok=True)
    os.makedirs(carpeta_gatos, exist_ok=True)
    
    print(f"📁 Creando estructura en: {ruta_destino}/")
    print(f"   ├── perro/")
    print(f"   └── gato/\n")
    
    # Obtener lista de archivos
    archivos = [f for f in os.listdir(ruta_original) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"📊 Archivos encontrados: {len(archivos)}\n")
    
    # Contadores
    contador_perros = 0
    contador_gatos = 0
    errores = 0
    
    # Procesar cada archivo
    print("🔄 Procesando imágenes...\n")
    
    for i, archivo in enumerate(archivos):
        try:
            ruta_origen = os.path.join(ruta_original, archivo)
            
            # Determinar si es perro o gato por el nombre del archivo
            if archivo.startswith('dog.'):
                # Verificar límite
                if limite_por_clase and contador_perros >= limite_por_clase:
                    continue
                
                # Nuevo nombre más limpio
                nuevo_nombre = f"perro_{contador_perros:05d}.jpg"
                ruta_destino_archivo = os.path.join(carpeta_perros, nuevo_nombre)
                
                # Copiar archivo
                shutil.copy2(ruta_origen, ruta_destino_archivo)
                contador_perros += 1
                
            elif archivo.startswith('cat.'):
                # Verificar límite
                if limite_por_clase and contador_gatos >= limite_por_clase:
                    continue
                
                # Nuevo nombre más limpio
                nuevo_nombre = f"gato_{contador_gatos:05d}.jpg"
                ruta_destino_archivo = os.path.join(carpeta_gatos, nuevo_nombre)
                
                # Copiar archivo
                shutil.copy2(ruta_origen, ruta_destino_archivo)
                contador_gatos += 1
            
            # Mostrar progreso cada 1000 imágenes
            if (i + 1) % 1000 == 0:
                print(f"  Procesadas: {i+1}/{len(archivos)} imágenes...")
                
        except Exception as e:
            errores += 1
            print(f"  ⚠️  Error con {archivo}: {e}")
    
    # Resumen final
    print("\n" + "="*70)
    print("  ✅ REORGANIZACIÓN COMPLETADA")
    print("="*70)
    print(f"\n📊 Resumen:")
    print("─"*70)
    print(f"  Perros copiados:  {contador_perros:>6} imágenes")
    print(f"  Gatos copiados:   {contador_gatos:>6} imágenes")
    print(f"  Total:            {contador_perros + contador_gatos:>6} imágenes")
    if errores > 0:
        print(f"  Errores:          {errores:>6}")
    print("─"*70)
    print(f"\n📁 Dataset organizado en: {ruta_destino}/")
    print("\n💡 Ahora puedes entrenar tu CNN:")
    print(f"   1. En main.py, usa: ruta_dataset = '{ruta_destino}/'")
    print(f"   2. Ejecuta: python main.py")
    print("="*70 + "\n")
    
    return True


def organizar_con_opciones():
    """
    Menú interactivo para organizar el dataset.
    """
    print("\n" + "="*70)
    print("  ORGANIZADOR DE DOGS VS CATS - KAGGLE")
    print("="*70 + "\n")
    
    print("Este script reorganizará tu dataset de Kaggle en la estructura:")
    print("  dataset/")
    print("    ├── perro/")
    print("    │   ├── perro_00000.jpg")
    print("    │   ├── perro_00001.jpg")
    print("    │   └── ...")
    print("    └── gato/")
    print("        ├── gato_00000.jpg")
    print("        ├── gato_00001.jpg")
    print("        └── ...\n")
    
    print("─"*70 + "\n")
    
    # Solicitar ruta
    print("📂 ¿Dónde está la carpeta 'train' de Dogs vs Cats?")
    print("   Ejemplos:")
    print("   • train")
    print("   • dogs-vs-cats/train")
    print("   • C:/Users/user/Downloads/dogs-vs-cats/train\n")
    
    ruta = input("Ruta: ").strip().strip('"').strip("'")
    
    if not ruta:
        ruta = "train"  # Valor por defecto
    
    # Verificar si existe
    if not os.path.exists(ruta):
        print(f"\n❌ Error: No se encuentra '{ruta}'")
        print("\nAsegúrate de:")
        print("  1. Haber descomprimido el archivo de Kaggle")
        print("  2. Estar en el directorio correcto")
        return
    
    # Preguntar por límite
    print("\n─"*70)
    print("\n⚙️  ¿Quieres limitar la cantidad de imágenes? (útil para pruebas)")
    print("   • Presiona ENTER para usar todas (~25,000 imágenes)")
    print("   • O escribe un número (ej: 500 para 500 perros + 500 gatos)\n")
    
    limite = input("Límite por clase (ENTER = todas): ").strip()
    
    if limite:
        try:
            limite = int(limite)
            print(f"\n✓ Se copiarán {limite} imágenes de cada clase")
        except:
            print("\n⚠️  Valor inválido, se usarán todas las imágenes")
            limite = None
    else:
        limite = None
        print("\n✓ Se copiarán todas las imágenes")
    
    # Confirmar
    print("\n─"*70)
    print("\n⚠️  Este proceso copiará las imágenes (no las moverá)")
    print("   Los archivos originales permanecerán intactos.\n")
    
    confirmar = input("¿Continuar? (s/n): ").strip().lower()
    
    if confirmar in ['s', 'si', 'y', 'yes']:
        print()
        organizar_dogs_vs_cats(ruta, "dataset", limite)
    else:
        print("\nOperación cancelada.\n")


if __name__ == "__main__":
    print("\n🐶🐱 Organizador de Dogs vs Cats para tu CNN\n")
    
    # Verificar si ya existe una estructura organizada
    if os.path.exists("dataset/perro") and os.path.exists("dataset/gato"):
        print("⚠️  Ya existe una carpeta 'dataset/' organizada.\n")
        respuesta = input("¿Quieres reorganizar de todos modos? (s/n): ").strip().lower()
        if respuesta not in ['s', 'si', 'y', 'yes']:
            print("\nUsando dataset existente. ¡Listo para entrenar!\n")
            exit(0)
    
    organizar_con_opciones()