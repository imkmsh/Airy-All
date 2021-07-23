import numpy as np


class LossOfNN:

    @staticmethod
    def sigmoid(z):
        activation_cache = [z]
        a = 1 / (1 + np.exp(-z))
        return a, activation_cache

    @staticmethod
    def relu(z):
        activation_cache = [z]
        a = np.maximum(0, z)
        return a, activation_cache

    @staticmethod
    def softmax(z):
        activation_cache = [z]
        z = z - np.max(z, axis=1, keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return a, activation_cache
