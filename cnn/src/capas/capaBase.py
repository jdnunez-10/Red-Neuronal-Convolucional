class Capa:
    def forward(self, entrada):
        """
        Calcula la salida de la capa durante el forward pass.
        """
        raise NotImplementedError("Forward no implementado")

    def backward(self, gradiente_salida, tasa_aprendizaje):
        """
        Calcula gradientes durante backpropagation y actualiza parámetros si los hay.
        """
        raise NotImplementedError("Backward no implementado")
