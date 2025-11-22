import numpy as np

class EntrenadorCNN:
    """
    Controlador del entrenamiento de la red neuronal.
    Implementa forward, backward y actualización de pesos.
    """

    def __init__(self, modelo, lr, funcion_perdida, derivada_funcion_perdida):
        self.modelo = modelo
        self.lr = lr
        self.funcion_perdida = funcion_perdida
        self.derivada_perdida = derivada_funcion_perdida
        
        # Historial de métricas
        self.historial = {
            'perdida': [],
            'accuracy': []
        }

    def entrenar(self, X, y, epochs=5, lote_size=8, validacion=None):
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Imágenes normalizadas con forma (N, C, H, W)
        y : numpy.ndarray
            Etiquetas enteras con forma (N,)
        epochs : int
            Número de épocas de entrenamiento
        lote_size : int
            Tamaño del batch
        validacion : tuple (X_val, y_val), opcional
            Datos de validación para evaluar después de cada época
        """

        num_muestras = X.shape[0]
        num_lotes = num_muestras // lote_size

        print(f"\n{'='*60}")
        print(f"  INICIO DEL ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"Muestras de entrenamiento: {num_muestras}")
        print(f"Tamaño de lote: {lote_size}")
        print(f"Número de lotes por época: {num_lotes}")
        print(f"Tasa de aprendizaje: {self.lr}")
        print(f"{'='*60}\n")

        for epoca in range(epochs):
            print(f"{'='*60}")
            print(f"  ÉPOCA {epoca+1}/{epochs}")
            print(f"{'='*60}")

            # Mezclar dataset cada época
            indices = np.arange(num_muestras)
            np.random.shuffle(indices)
            X_mezclado = X[indices]
            y_mezclado = y[indices]

            perdida_total = 0
            accuracy_total = 0
            lotes_procesados = 0

            # Entrenamiento por lotes
            for i in range(0, num_muestras, lote_size):
                X_lote = X_mezclado[i:i+lote_size]
                y_lote = y_mezclado[i:i+lote_size]

                # Validar que el lote no esté vacío
                if len(X_lote) == 0:
                    continue

                # Forward completo
                logits = self.modelo.forward(X_lote)

                # Calcular pérdida
                perdida = self.funcion_perdida(logits, y_lote)
                perdida_total += perdida

                # Calcular accuracy del lote
                predicciones = np.argmax(logits, axis=1)
                accuracy_lote = np.mean(predicciones == y_lote)
                accuracy_total += accuracy_lote

                # Calcular gradiente de la pérdida
                gradiente = self.derivada_perdida(logits, y_lote)

                # Backpropagation a través del modelo
                self.modelo.backward(gradiente, self.lr)

                lotes_procesados += 1

                # Mostrar progreso cada 10 lotes
                if (lotes_procesados % 10 == 0) or (lotes_procesados == num_lotes):
                    print(f"  Lote {lotes_procesados}/{num_lotes} - "
                          f"Pérdida: {perdida:.4f} - "
                          f"Accuracy: {accuracy_lote*100:.2f}%")

            # Promedios de la época
            perdida_promedio = perdida_total / lotes_procesados
            accuracy_promedio = accuracy_total / lotes_procesados

            # Guardar en historial
            self.historial['perdida'].append(perdida_promedio)
            self.historial['accuracy'].append(accuracy_promedio)

            print(f"\n{'─'*60}")
            print(f"  RESUMEN ÉPOCA {epoca+1}")
            print(f"{'─'*60}")
            print(f"  Pérdida promedio: {perdida_promedio:.4f}")
            print(f"  Accuracy promedio: {accuracy_promedio*100:.2f}%")

            # Evaluación en validación si está disponible
            if validacion is not None:
                X_val, y_val = validacion
                val_accuracy = self.evaluar(X_val, y_val, lote_size)
                print(f"  Accuracy validación: {val_accuracy*100:.2f}%")

            print(f"{'─'*60}\n")

        print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}\n")

    def evaluar(self, X, y, lote_size=8):
        """
        Evalúa el modelo en un conjunto de datos.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Imágenes de validación/prueba
        y : numpy.ndarray
            Etiquetas verdaderas
        lote_size : int
            Tamaño del batch para evaluación
        
        Retorna:
        --------
        float
            Accuracy del modelo
        """
        num_muestras = X.shape[0]
        predicciones_correctas = 0

        for i in range(0, num_muestras, lote_size):
            X_lote = X[i:i+lote_size]
            y_lote = y[i:i+lote_size]

            # Forward pass
            logits = self.modelo.forward(X_lote)
            predicciones = np.argmax(logits, axis=1)

            # Contar aciertos
            predicciones_correctas += np.sum(predicciones == y_lote)

        accuracy = predicciones_correctas / num_muestras
        return accuracy

    def mostrar_historial(self):
        """
        Muestra el historial de entrenamiento.
        """
        print(f"\n{'='*60}")
        print(f"  HISTORIAL DE ENTRENAMIENTO")
        print(f"{'='*60}")
        for i, (perdida, accuracy) in enumerate(zip(self.historial['perdida'], 
                                                     self.historial['accuracy'])):
            print(f"Época {i+1}: Pérdida = {perdida:.4f}, Accuracy = {accuracy*100:.2f}%")
        print(f"{'='*60}\n")