import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


@tf.function

class Data:

    def __init__(self):
        self.result = 0

    def make_gaussian_x_data(self, num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov):
        negative_samples = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov, size=num_of_samples)
        positive_samples = np.random.multivariate_normal(mean=positive_mean, cov=positive_cov, size=num_of_samples)

        x = np.vstack((negative_samples, positive_samples)).astype(np.float32)

        return x

    def make_gaussian_y_data(self, num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov):
        negative_samples = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov, size=num_of_samples)
        positive_samples = np.random.multivariate_normal(mean=positive_mean, cov=positive_cov, size=num_of_samples)

        y = np.vstack((np.zeros((num_of_samples, 1), dtype='float32'),
                       np.ones((num_of_samples, 1), dtype='float32')))

        return y

    def graph(self, x, y):
        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
        plt.show()


class Neuron:

    def __init__(self):
        self.result = 0

    def __call__(self, dataset, x_test, y_test, learning_rate, num_of_iterations, num_of_epochs, loss_function):
        dim = x_test.shape[1]
        params = parameter.init_params(dim)
        losses = []
        for epoch in range(num_of_epochs):
            batch_dataset = dataset.shuffle(buffer_size=1000).batch(1000)
            for i, (x, y) in enumerate(batch_dataset):
                for j in range(num_of_iterations):
                    m = x.shape[0]
                    grads, loss = self.forward_and_backward(x, y, m, params["w"], params["b"], loss_function)
                    params = self.update_params(params, learning_rate, grads)
                    if j % 100 == 0:
                        losses.append(loss)
                        print(f"epoch: {epoch}, loss after iteration {j}:  {loss}")
                        train_accuracy = self.predict(params, x, y)
                        test_accuracy = self.predict(params, x_test, y_test)
                        print(f"train accuracy: {train_accuracy}%")
                        print(f"test accuracy: {test_accuracy}%")

        train_accuracy = self.predict(params, x_train, y_train)
        test_accuracy = self.predict(params, x_test, y_test)
        print(f"final train accuracy: {train_accuracy}%")
        print(f"final test accuracy: {test_accuracy}%")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("num of iterations")
        plt.ylabel("loss")
        plt.title("logistic logistic_regression")
        plt.show()

        return params

    def init_params(dim):
        params = {"w": tf.Variable(tf.random.normal(shape=(dim, 1))),
                  "b": tf.Variable(tf.zeros(shape=(1,)))}
        return params

    def update_params(params, learning_rate, grads):
        params["w"].assign_sub(learning_rate * grads[0])
        params["b"].assign_sub(learning_rate * grads[1])
        return params

    def forward(self, x, w, b):
        z = tf.linalg.matmul(x, w) + b
        a = tf.math.sigmoid(z)
        return a

    def calc_loss(self, y, a, m, loss_function, w):
        if loss_function == "cross_entropy": i
        return tf.math.reduce_sum(-(y * tf.math.log(a) + (1 - y) * tf.math.log(1 - a))) / m + 0.1 * tf.reduce_sum(tf.math.square(w)) / 2

        elif loss_function == "mean_square_error":
        return tf.math.reduce_sum(tf.math.square(y - a)) / m

    def predict(self, params, x, y):
        w = params["w"]
        b = params["b"]
        a = move.forward(x, w, b)
        prediction = np.zeros(a.shape)
        prediction[a >= 0.9] = 1
        accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
        return accuracy

    def forward_and_backward(self, x, y, m, w, b, loss_function):
        with tf.GradientTape() as tape:
            a = self.forward(x, w, b)
            loss = self.calc_loss(y, a, m, loss_function, w)
            grads = tape.gradient(loss, [w, b])
        return grads, loss




x_train = data.make_gaussian_x_data(10000, negative_mean=[3.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                      positive_mean=[1.0, 3.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]])

y_train = data.make_gaussian_y_data(10000, negative_mean=[3.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                      positive_mean=[1.0, 3.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]])

data.graph(x_train, y_train)

x_test = data.make_gaussian_x_data(1000, negative_mean=[2.0, -1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                    positive_mean=[-2.0, 2.0], positive_cov=[[4.0, 1.0], [1.0, 2.0]])

y_test = data.make_gaussian_y_data(1000, negative_mean=[2.0, -1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                    positive_mean=[-2.0, 2.0], positive_cov=[[4.0, 1.0], [1.0, 2.0]])

data.graph(x_test, y_test)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

model.logistic_regression_model(dataset, x_test, y_test,
                          learning_rate=0.001,
                          num_of_iterations=1000,
                          num_of_epochs=5,
                          loss_function="cross_entropy")
