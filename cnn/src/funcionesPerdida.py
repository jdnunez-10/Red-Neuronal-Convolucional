import numpy as np

def funcion_perdida_crossentropy(logits, y):
    """
    Calcula la pérdida de entropía cruzada desde logits.
    
    Parámetros:
    -----------
    logits : numpy.ndarray
        Salida de la red sin softmax, forma (N, num_clases)
    y : numpy.ndarray
        Etiquetas enteras de las clases correctas, forma (N,)
    
    Retorna:
    --------
    float
        Pérdida promedio del batch
    """
    N = logits.shape[0]
    
    # Aplicar softmax para estabilidad numérica
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Extraer las probabilidades de las clases correctas
    correct_probs = probs[np.arange(N), y]
    
    # Calcular pérdida (agregar epsilon para evitar log(0))
    loss = -np.mean(np.log(correct_probs + 1e-15))
    
    return loss


def derivada_crossentropy(logits, y):
    """
    Calcula el gradiente de la pérdida de entropía cruzada 
    con respecto a los logits (incluye softmax).
    
    Parámetros:
    -----------
    logits : numpy.ndarray
        Salida de la red sin softmax, forma (N, num_clases)
    y : numpy.ndarray
        Etiquetas enteras de las clases correctas, forma (N,)
    
    Retorna:
    --------
    numpy.ndarray
        Gradiente con forma (N, num_clases)
    """
    N = logits.shape[0]
    
    # Aplicar softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # El gradiente de cross-entropy + softmax es simplemente: probs - one_hot(y)
    grad = probs.copy()
    grad[np.arange(N), y] -= 1
    
    # Normalizar por tamaño del batch
    grad /= N
    
    return grad


# --------------------------------------------------
#  CLASE ALTERNATIVA (Si prefieres usar OOP)
# --------------------------------------------------
class EntropiaCruzada:
    """
    Implementación orientada a objetos de la pérdida Cross-Entropy
    usada para clasificación multiclase.
    """

    def calcular(self, logits, y):
        """
        Calcula la pérdida.
        
        Parámetros:
        -----------
        logits : numpy.ndarray
            Salida de la red (N, num_clases)
        y : numpy.ndarray
            Etiquetas enteras (N,)
        """
        return funcion_perdida_crossentropy(logits, y)

    def gradiente(self, logits, y):
        """
        Calcula el gradiente de la pérdida.
        
        Parámetros:
        -----------
        logits : numpy.ndarray
            Salida de la red (N, num_clases)
        y : numpy.ndarray
            Etiquetas enteras (N,)
        """
        return derivada_crossentropy(logits, y)