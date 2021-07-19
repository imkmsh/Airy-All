import random
import numpy as np
import matplotlib.pyplot as plt

# parameter 초기화하는 함수 정의, w와 b 모양 잡아주기
def init_params(dim):
    params = {"w": np.random.randn(dim, 1) * 0.001, "b": 0}
    # 평균이 0이고 표준이 1인 가우시안 랜덤으로 matrix array[dim, 1} 생성, 수가 너무 커지는 것을 방지하고자 0.001을 곱해준다. b는 0으로 초기화.
    return params

# 선형변환 (선형함수)
def linear(x, w, b):
    linear_cache = [x, w, b]
    # 나중에도 필요하니까 cache로 저장
    z = np.matmul(x, w) + b
    # z로 저장
    return z, linear_cache


def linear_grad(dz, linear_cache):
    [x, w, b] = linear_cache
    grads = {"dw": np.matmul(x.T, dz),
             "db": np.mean(dz)}
    return grads

# 비선형변환 (활성함수)
def sigmoid(z):
    activation_cache = [z]
    a = 1 / (1 + np.exp(-z))
    return a, activation_cache

def sigmoid_grad(dloss, activation_cache):
    # L 함수 전체 w로 미분 (dL/da)(da/dz)(dz/dw)
    [z] = activation_cache
    a = 1 / (1 + np.exp(-z))
    return dloss * a * (1 - a)

def forward(x, w, b):
    z, linear_cache = linear(x, w, b)
    # 선형변환하고
    a, activation_cache = sigmoid(z)
    # 시그모이드 돌리고
    return a, linear_cache, activation_cache

def calc_loss(y, a, m, loss_function, w, lamb):
    if loss_function == "cross_entropy":
        # 베르누이 분포로 모델링한다면
        return np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m + lamb * np.sum(np.square(w)) / 2
    elif loss_function == "mean_square_error":
        # 평균_제곱근_편차
        return np.sum(np.square(y - a)) / (2 * m)

def loss_grad(y, a, m, loss_function):
    # loss를 미분
    if loss_function == "cross_entropy":
        return -(y / a - (1 - y) / (1 - a)) / m
    elif loss_function == "mean_square_error":
        return (a - y) / m

# forward, calc_loss, loss_grad, sigmoid_grad, linear_grad : dw, db를 구한다 => backward
def backward(y, a, m, linear_cache, activation_cache, loss_function):
    dloss = loss_grad(y, a, m, loss_function)
    dz = sigmoid_grad(dloss, activation_cache)
    grads = linear_grad(dz, linear_cache)
    return grads

# for/backward 한꺼번에
def forward_and_backward(x, y, m, w, b, loss_function, lamb):
    a, linear_cache, activation_cache = forward(x, w, b)
    loss = calc_loss(y, a, m, loss_function, w, lamb)
    grads = backward(y, a, m, linear_cache, activation_cache, loss_function)
    return loss, grads
# loss는 왜 리턴할까 -> 나중에 잘 되고 있는지 출력하면서 확인하면서 하려고

# w, b update
def update_params(params, learning_rate, grads, lamb):
    params["w"] = params["w"] * (1 - lamb) - learning_rate * grads["dw"]
    params["b"] = params["b"] - learning_rate * grads["db"]
    return params

def predict(params, x, y):
    w = params["w"]
    b = params["b"]
    a, _, _ = forward(x, w, b)
    prediction = np.zeros(a.shape)
    prediction[a >= 0.9] = 1
    accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
    return accuracy

def make_gaussian_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov):
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

def logistic_regression_model(x_train, y_train, x_test, y_test,
                              learning_rate, num_of_iterations, loss_function, lamb):
    dim = x_train.shape[1]
    params = init_params(dim)
    losses = []
    for i in range(num_of_iterations):
        m = x_train.shape[0]
        loss, grads = forward_and_backward(x_train, y_train, m, params["w"], params["b"], loss_function, lamb)
        params = update_params(params, learning_rate, grads, lamb)
        if i % 100 == 0:
            losses.append(loss)
            print(f"loss after iteration {i}:  {loss}")
            train_accuracy = predict(params, x_train, y_train)
            test_accuracy = predict(params, x_test, y_test)
            print(f"train accuracy: {train_accuracy}%")
            print(f"test accuracy: {test_accuracy}%")

    train_accuracy = predict(params, x_train, y_train)
    test_accuracy = predict(params, x_test, y_test)
    print(f"final train accuracy: {train_accuracy}%")
    print(f"final test accuracy: {test_accuracy}%")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("num of iterations")
    plt.ylabel("loss")
    plt.title("logistic regression")
    plt.show()

    return params

x_train, y_train = make_gaussian_data(100000, negative_mean=[3.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                      positive_mean=[1.0, 3.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]])
x_test, y_test = make_gaussian_data(100, negative_mean=[2.0, -1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                    positive_mean=[-2.0, 2.0], positive_cov=[[4.0, 1.0], [1.0, 2.0]])
logistic_regression_model(x_train, y_train, x_test, y_test,
                          learning_rate=0.1,
                          num_of_iterations=5000,
                          loss_function="cross_entropy",
                          lamb=0.01)