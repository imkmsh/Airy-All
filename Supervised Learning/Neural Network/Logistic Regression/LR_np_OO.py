class DataOfML:
    def __init__(self):
        self.result = 0

    def __call__(self):
        return 0

    @staticmethod
    def make_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov):
        negative_samples = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov, size=num_of_samples)
        positive_samples = np.random.multivariate_normal(mean=positive_mean, cov=positive_cov, size=num_of_samples)

        x = np.vstack((negative_samples, positive_samples)).astype(np.float32)
        y = np.vstack((np.zeros((num_of_samples, 1), dtype='float32'),
                       np.ones((num_of_samples, 1), dtype='float32')))

        plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
        plt.show()
        print(x)
        print(y)
        return x, y


class Neuron:
    def __init__(self, learning_rate, lamb):
        self.params = {"w": 0, "b": 0}
        self.learning_rate = learning_rate
        self.lamb = lamb

    def __call__(self, x_train, y_train, x_test, y_test, num_of_iterations, loss_function):
        self.params = {"w": np.random.randn(x_train.shape[1], 1) * 0.001, "b": 0}
        losses = []
        for i in range(num_of_iterations):
            m = x_train.shape[0]
            loss, grads = self.forward_and_backward(x_train, y_train, m, self.params["w"], self.params["b"],
                                                    loss_function, self.lamb)
            params = self.update_params(self.params, self.learning_rate, grads, self.lamb)
            if i % 100 == 0:
                losses.append(loss)
                print(f"loss after iteration {i}:  {loss}")
                train_accuracy = self.predict(self.params, x_train, y_train)
                test_accuracy = self.predict(self.params, x_test, y_test)
                print(f"train accuracy: {train_accuracy}%")
                print(f"test accuracy: {test_accuracy}%")

        train_accuracy = self.predict(self.params, x_train, y_train)
        test_accuracy = self.predict(self.params, x_test, y_test)
        print(f"final train accuracy: {train_accuracy}%")
        print(f"final test accuracy: {test_accuracy}%")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("num of iterations")
        plt.ylabel("loss")
        plt.title("logistic regression")
        plt.show()
        return self.params

    def forward(self, x, w, b):
        z, linear_cache = MathOfNN.linear(x, w, b)
        # 선형변환하고
        a, activation_cache = MathOfNN.sigmoid(z)
        # 시그모이드 돌리고
        return a, linear_cache, activation_cache

    def calc_loss(self, y, a, m, loss_function, w, lamb):
        if loss_function == "cross_entropy":
            # 베르누이 분포로 모델링한다면
            return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m + lamb * np.sum(np.square(w)) / 2
        elif loss_function == "mean_square_error":
            # 평균_제곱근_편차
            return np.sum(np.square(y - a)) / (2 * m)

    def loss_grad(self, y, a, m, loss_function):
        # loss를 미분
        if loss_function == "cross_entropy":
            return MathOfNN.cross_entropy_gradient(y, a, m)
        elif loss_function == "mean_square_error":
            return MathOfNN.mean_square_error_gradient(y, a, m)

    # forward, calc_loss, loss_grad, sigmoid_grad, linear_grad : dw, db를 구한다 => backward
    def backward(self, y, a, m, linear_cache, activation_cache, loss_function):
        dloss = self.loss_grad(y, a, m, loss_function)
        dz = MathOfNN.sigmoid_grad(dloss, activation_cache)
        grads = MathOfNN.linear_grad(dz, linear_cache)
        return grads

    # for/backward 한꺼번에
    def forward_and_backward(self, x, y, m, w, b, loss_function, lamb):
        a, linear_cache, activation_cache = self.forward(x, w, b)
        loss = self.calc_loss(y, a, m, loss_function, w, lamb)
        grads = self.backward(y, a, m, linear_cache, activation_cache, loss_function)
        return loss, grads

    # loss는 왜 리턴할까 -> 나중에 잘 되고 있는지 출력하면서 확인하면서 하려고

    # w, b update
    def update_params(self, params, learning_rate, grads, lamb):
        params["w"] = params["w"] * (1 - lamb) - learning_rate * grads["dw"]
        params["b"] = params["b"] - learning_rate * grads["db"]
        return params

    def predict(self, params, x, y):
        w = params["w"]
        b = params["b"]
        a, _, _ = self.forward(x, w, b)
        prediction = np.zeros(a.shape)
        prediction[a >= 0.9] = 1
        accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
        return accuracy


if __name__ == "__main__":
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from ..util.math import MathOfNN

    x_train, y_train = DataOfML.make_data(5000, negative_mean=[10.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                          positive_mean=[1.0, 10.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]])

    x_test, y_test = DataOfML.make_data(100, negative_mean=[8.0, -1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                        positive_mean=[-1.0, 4.0], positive_cov=[[4.0, 1.0], [1.0, 2.0]])

    model = Neuron(0.1, 0.01)
    model(x_train, y_train, x_test, y_test, num_of_iterations=5000,
          loss_function="cross_entropy")
