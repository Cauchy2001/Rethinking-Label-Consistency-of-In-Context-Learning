import numpy as np


def entropy(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        # print(probs)
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p)
        # print(entropy)
        return entropy
    
    # print(probs)
    return - (mask * probs * np.log(probs)).sum(label_dim)
    
def probs_argmax(probs: np.array, label_dim: int = 0, mask=None):
    if mask is None:
        return np.argmax(probs , axis=label_dim)
    return np.argmax(mask * probs , axis=label_dim)
