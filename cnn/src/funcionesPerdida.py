import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def funcion_perdida_crossentropy(logits, y):
    """
    Cross-Entropy Loss desde logits y etiquetas enteras.
    """
    N = logits.shape[0]
    probs = softmax(logits)
    correct_probs = probs[np.arange(N), y]
    loss = -np.mean(np.log(correct_probs + 1e-15))
    return loss

def derivada_crossentropy(logits, y):
    """
    Gradiente de Cross-Entropy + Softmax respecto a logits.
    """
    N = logits.shape[0]
    probs = softmax(logits)
    probs[np.arange(N), y] -= 1
    return probs / N
