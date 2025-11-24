import numpy as np

class EntrenadorCNN:
    def __init__(self, modelo, lr, funcion_perdida, derivada_funcion_perdida):
        self.modelo = modelo
        self.lr = lr
        self.funcion_perdida = funcion_perdida
        self.derivada_perdida = derivada_funcion_perdida

        self.historial = {
            'perdida': [],
            'accuracy': [],
            'val_accuracy': []
        }

    def entrenar(self, X, y, epochs=15, lote_size=32, validacion=None):

        # Normalizar imágenes desde aquí
        X = X.astype("float32") / 255.0
        if validacion:
            X_val = validacion[0].astype("float32") / 255.0
            y_val = validacion[1]

        num_muestras = X.shape[0]
        num_lotes = num_muestras // lote_size

        for epoca in range(epochs):
            print(f"\n---- ÉPOCA {epoca+1}/{epochs} ----")

            indices = np.random.permutation(num_muestras)
            X_mezclado = X[indices]
            y_mezclado = y[indices]

            perdida_total = 0
            correctos = 0
            total = 0

            for i in range(0, num_muestras, lote_size):
                X_lote = X_mezclado[i:i+lote_size]
                y_lote = y_mezclado[i:i+lote_size]

                if X_lote.shape[0] < lote_size:
                    continue

                # Mezcla interna del lote
                idx = np.random.permutation(X_lote.shape[0])
                X_lote = X_lote[idx]
                y_lote = y_lote[idx]

                logits = self.modelo.forward(X_lote)

                # pérdida
                perdida = self.funcion_perdida(logits, y_lote)
                perdida_total += perdida

                # accuracy
                pred = np.argmax(logits, axis=1)
                correctos += np.sum(pred == y_lote)
                total += len(y_lote)

                # backprop
                grad = self.derivada_perdida(logits, y_lote)

                # Clip gradiente para estabilidad
                grad = np.clip(grad, -1.0, 1.0)
                # Backpropagation - 
                self.modelo.backward(grad, self.lr)
                
                # Actualizar pesos -
                self.modelo.actualizar()

            # Métricas de la época
            perdida_promedio = perdida_total / num_lotes
            accuracy_promedio = correctos / total

            self.historial['perdida'].append(perdida_promedio)
            self.historial['accuracy'].append(accuracy_promedio)

            print(f" Pérdida: {perdida_promedio:.4f}")
            print(f" Accuracy: {accuracy_promedio*100:.2f}%")

            if validacion is not None:
                X_val, y_val = validacion
            val_accuracy = self.evaluar(X_val, y_val, lote_size)
            self.historial['val_accuracy'].append(val_accuracy)
            print(f"  Accuracy validación: {val_accuracy*100:.2f}%")

            print(f"{'─'*60}\n")

            print(f"\n{'='*60}")
        print(f"  ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}\n")


    def evaluar(self, X, y, lote_size=32):
        X = X.astype("float32") / 255.0

        correctos = 0
        total = 0

        for i in range(0, X.shape[0], lote_size):
            logits = self.modelo.forward(X[i:i+lote_size])
            pred = np.argmax(logits, axis=1)
            correctos += np.sum(pred == y[i:i+lote_size])
            total += len(y[i:i+lote_size])

        return correctos / total


def mostrar_historial(self):
        print(f"\n{'='*60}")
        print(f"  HISTORIAL DE ENTRENAMIENTO")
        print(f"{'='*60}")
        for i, (perdida, accuracy) in enumerate(zip(self.historial['perdida'], 
                                                     self.historial['accuracy'])):
            val_acc = self.historial['val_accuracy'][i] if i < len(self.historial['val_accuracy']) else None
            resumen = f"Época {i+1}: Pérdida = {perdida:.4f}, Accuracy = {accuracy*100:.2f}%"
            if val_acc is not None:
                resumen += f", Val Accuracy = {val_acc*100:.2f}%"
            print(resumen)
        print(f"{'='*60}\n")