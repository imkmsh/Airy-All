class DataOfDNN:
    def __init__(self):
        self.result = 0

    def __call__(self):
        return 0

    @staticmethod
    def make_gaussian_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov, negative_mean2,
                           positive_mean2):
        half_num_of_samples = int(num_of_samples / 2)
        negative_samples1 = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov,
                                                          size=half_num_of_samples)
        negative_samples2 = np.random.multivariate_normal(mean=positive_mean, cov=negative_cov,
                                                          size=half_num_of_samples)
        positive_samples1 = np.random.multivariate_normal(mean=negative_mean2, cov=positive_cov,
                                                          size=half_num_of_samples)
        positive_samples2 = np.random.multivariate_normal(mean=positive_mean2, cov=positive_cov,
                                                          size=half_num_of_samples)

        x = np.vstack((negative_samples1, negative_samples2, positive_samples1, positive_samples2)).astype(np.float32)
        y = np.vstack((np.zeros((num_of_samples, 1), dtype='float32'),
                       np.ones((num_of_samples, 1), dtype='float32')))

        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
        plt.show()

        return x, y


class Neuron:
    def __init__(self, num_of_layers, activation, last_activation, learning_rate, lamb):
        self.params = {}
        self.activation = activation
        self.last_activation = last_activation
        self.num_of_layers = num_of_layers
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.m = 0

    def __call__(self, seq, x_train, y_train, x_test, y_test,
                 loss_function, dim_of_layer1, dim_of_layer2):
        self.params["w" + str(seq)] = np.random.randn(dim_of_layer1, dim_of_layer2)
        self.params["b" + str(seq)] = np.zeros((1, dim_of_layer2))
        self.seq = seq
        self.m = x_train.shape[0]
        a, grads, self.loss = self.forward_and_backward(x_train, y_train, self.m, loss_function)
        self.params = self.update_params(self.params, grads)
        return a

    def single_forward(self, a, w, b):
        z, linear_cache = MathOfNN.linear(a, w, b)
        activation_cache = 0
        if self.activation == "relu":
            a, activation_cache = MathOfNN.relu(z)
        elif self.activation == "leaky_relu":
            a, activation_cache = MathOfNN.leaky_relu(z)
        elif self.activation == "sigmoid":
            a, activation_cache = MathOfNN.sigmoid(z)
        elif self.activation == "softmax":
            a, activation_cache = MathOfNN.softmax(z)
        cache = linear_cache, activation_cache
        return a, cache

    def forward(self, x):
        a = x
        caches = []
        if self.seq != self.num_of_layers - 1:
            a, cache = self.single_forward(a, self.params['w' + str(self.seq)], self.params['b' + str(self.seq)],
                                           )
        else:
            a, cache = self.single_forward(a, self.params['w' + str(self.seq)], self.params['b' + str(self.seq)],
                                           )
        caches.append(cache)
        return a, caches

    def calc_loss(self, y, a, m, loss_function):
        if loss_function == "cross_entropy":
            return np.sum(-(y * np.log(a))) / self.m
        elif loss_function == "mean_square_error":
            return np.sum(np.square(y - a)) / (2 * self.m)

    def loss_grad(self, y, a, loss_function):
        if loss_function == "cross_entropy":
            return MathOfNN.cross_entropy_grad(y, a, self.m)
        elif loss_function == "mean_square_error":
            return MathOfNN.mean_square_error_grad(y, a, self.m)

    def single_backward(self, da, cache):
        linear_cache, activation_cache = cache
        dz = 0
        if self.activation == "sigmoid":
            dz = MathOfNN.sigmoid_grad(da, activation_cache)
        elif self.activation == "relu":
            dz = MathOfNN.relu_gradient(da, activation_cache)
        elif self.activation == "leaky_relu":
            dz = MathOfNN.leaky_relu_gradient(da, activation_cache)
        elif self.activation == "softmax":
            dz = MathOfNN.softmax_gradient(da, activation_cache)
        grads = MathOfNN.linear_grad(dz, linear_cache)
        return grads['dw'], grads['db'], grads['da']

    def backward(self, y, a, m, caches, loss_function):
        grads = {}
        dloss = self.loss_grad(y, a, loss_function)
        if self.seq == self.num_of_layers - 1:
            grads["dw" + str(self.seq)], grads["db" + str(self.seq)], grads[
                "da" + str(self.seq - 1)] = self.single_backward(dloss,
                                                                 caches[self.seq - 1])
        else:
            grads["dw" + str(self.seq)], grads["db" + str(self.seq)], grads[
                "da" + str(self.seq - 1)] = self.single_backward(
                grads["da" + str(self.seq)],
                caches[self.seq - 1])
        return grads

    def forward_and_backward(self, x, y, m, loss_function):
        a, caches = self.forward(x)
        loss = self.calc_loss(y, a, m, loss_function)
        grads = self.backward(y, a, m, caches, loss_function)
        return a, grads, loss

    def gradient_clip(self, grad, limit):
        if np.linalg.norm(grad) >= limit:
            grad = limit * (grad / np.linalg.norm(grad))
        return grad

    def update_params(self, params, grads):
        params["w" + str(self.seq)] = params["w" + str(self.seq)] * (
                1 - self.lamb * self.learning_rate) - self.learning_rate * self.gradient_clip(
            grads["dw" + str(self.seq)], 1)
        params["b" + str(self.seq)] = params["b" + str(self.seq)] - self.learning_rate * self.gradient_clip(
            grads["db" + str(self.seq)], 1)
        return params

    def predict(self, x, y):
        a, _ = self.forward(x)
        prediction = np.zeros(a.shape)
        prediction[a >= 0.8] = 1
        accuracy = np.mean(np.all(prediction == y, axis=1, keepdims=True)) * 100
        return accuracy


class DNN:
    def __init__(self, num_of_iterations, num_of_epochs):
        self.num_of_iterations = num_of_iterations
        self.num_of_epochs = num_of_epochs

    def __call__(self, num_of_layers, activation, last_activation, learning_rate, lamb):
        func = Neuron(num_of_layers, activation, last_activation, learning_rate, lamb)
        losses = []
        for epoch in range(self.num_of_epochs):
            for j in range(self.num_of_iterations):
                network()
                if j % 1000 == 0:
                    losses.append(func.loss)
                    print(f"epoch: {epoch + 1}, loss after iteration {j}: {func.loss}")
                    train_accuracy = func.predict(x_train, y_train)
                    test_accuracy = func.predict(x_test, y_test)
                    print(f"train accuracy: {train_accuracy}%")
                    print(f"test accuracy: {test_accuracy}%")

        train_accuracy = func.predict(x_train, y_train)
        test_accuracy = func.predict(x_test, y_test)
        print(f"final train accuracy: {train_accuracy}%")
        print(f"final test accuracy: {test_accuracy}%")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("num of iterations")
        plt.ylabel("loss")
        plt.title("deep neural network")
        plt.show()


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from ..util.math import MathOfNN

x_train, y_train = DataOfDNN.make_gaussian_data(1000, negative_mean=[1.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                                positive_mean=[20.0, 20.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]],
                                                negative_mean2=[1.0, 20.0], positive_mean2=[20.0, 1.0])
x_test, y_test = DataOfDNN.make_gaussian_data(100, negative_mean=[1.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                              positive_mean=[20.0, 20.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]],
                                              negative_mean2=[1.0, 20.0], positive_mean2=[20.0, 1.0])
dnn_process = Neuron(3, "relu", "sigmoid", 0.001, 0.1)
dnn_model = DNN(10000, 10)


def network():
    dnn_process(2, dnn_process(1, x_train, y_train, x_test, y_test,
                               loss_function="cross_entropy",
                               dim_of_layer1=2,
                               dim_of_layer2=8), y_train, x_test, y_test,
                loss_function="cross_entropy",
                dim_of_layer1=8,
                dim_of_layer2=1)


dnn_model(3, "relu", "sigmoid", 0.001, 0.1)
