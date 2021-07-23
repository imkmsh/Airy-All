import numpy as np


class MathOfNN:

    @staticmethod
    def linear(a, w, b):
        linear_cache = [a, w, b]
        z = np.matmul(a, w) + b
        return z, linear_cache

    @staticmethod
    def linear_grad(dz, linear_cache):
        [a, w] = linear_cache
        grads = {"dw": np.matmul(a.T, dz),
                 "db": np.mean(dz, axis=0, keepdims=True),
                 "da": np.matmul(dz, w.T)}
        return grads

    @staticmethod
    def sigmoid_grad(da, activation_cache):
        [z] = activation_cache
        a = 1 / (1 + np.exp(-z))
        return da * a * (1 - a)

    @staticmethod
    def relu_gradient(da, activation_cache):
        [z] = activation_cache
        dz = np.ones(z.shape)
        dz[z < 0] = 0
        return da * dz

    @staticmethod
    def leaky_relu(z):
        activation_cache = [z]
        a = np.maximum(0.01 * z, z)
        return a, activation_cache

    @staticmethod
    def leaky_relu_gradient(da, activation_cache):
        [z] = activation_cache
        dz = np.ones(da.shape)
        dz[z < 0] = 0.01
        return da * dz

    @staticmethod
    def softmax_gradient(dloss, activation_cache):
        [z] = activation_cache
        z = z - np.max(z, axis=1, keepdims=True)
        a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        dz = np.zeros(dloss.shape)
        (m, dim) = dloss.shape
        for k in range(m):
            middle_matrix = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        middle_matrix[i, j] = a[k, i] * (1 - a[k, i])
                    else:
                        middle_matrix[i, j] = -(a[k, i] * a[k, j])
            dz[k, :] = np.matmul(dloss[k, :], middle_matrix)
        return dz

    @staticmethod
    def cross_entropy_grad(y, a, m):
        return - (y / a) / m

    @staticmethod
    def mean_square_error_grad(y, a, m):
        return (a - y) / m
