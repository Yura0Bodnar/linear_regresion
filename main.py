import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

lambda_l1 = 0.001  # Регуляризаційний параметр L1
lambda_l2 = 0.01  # Регуляризаційний параметр L2
learning_rate = 0.01
n_iterations = 10000
eps = 1e-5


def compute_r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def mserror(X, w, y):
    y_pred = X.dot(w)
    return np.sum((y - y_pred) ** 2) / len(y_pred)


def gradient_mse(X, w, y):
    y_pred = X.dot(w)
    grad = 2 / len(X) * X.T.dot(y_pred - y)
    return grad


def gradient_mse_l1(X, w, y, lambda_l1):
    y_pred = X.dot(w)
    grad = 2/len(X) * X.T.dot(y_pred - y)
    # Додавання градієнтів штрафів L1 і L2
    l1_grad = lambda_l1 * np.sign(w)
    return grad + l1_grad


def gradient_mse_l2(X, w, y, lambda_l2):
    y_pred = X.dot(w)
    grad = 2/len(X) * X.T.dot(y_pred - y)
    # Додавання градієнтів штрафів L1 і L2
    l2_grad = 2 * lambda_l2 * w
    return grad + l2_grad


def solution(X_train, y_train, X_test, y_test, ind):
    mse = 0
    w = np.zeros(X_train.shape[1])
    for i in range(n_iterations):
        if ind == "l1":
            grad = gradient_mse_l1(X_train, w, y_train, lambda_l1)
        elif ind == "l2":
            grad = gradient_mse_l2(X_train, w, y_train, lambda_l2)
        else:
            grad = gradient_mse(X_train, w, y_train)
        new_w = w - learning_rate * grad
        mse = mserror(X_train, new_w, y_train)

        if i % 100 == 0:
            print(f"Iteration {i}: MSE = {mse}")

        if np.linalg.norm(new_w - w, ord=2) < eps:
            print('Convergence reached after', i+1, 'iterations.')
            break
        w = new_w

    y_pred_train = X_train.dot(w)
    r_squared_train = compute_r_squared(y_train, y_pred_train)

    print('Custom model coefficients:', w)
    print('Custom MSE train:', mse)
    print('Train R^2 Score:', r_squared_train)
    print('\n')


def main():
    data = pd.read_csv('adm_data.csv')
    X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
    y = data['Chance of Admit '].values

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Вбудована модель
    b_model = LinearRegression()
    b_model.fit(X_train, y_train)
    y_pred_b_test = b_model.predict(X_test)
    y_pred_b_train = b_model.predict(X_train)
    mse_b_test = mean_squared_error(y_test, y_pred_b_test)
    mse_b_train = mean_squared_error(y_train, y_pred_b_train)
    print('Sklearn MSE test:', mse_b_test)
    print('Sklearn MSE train:', mse_b_train)

    r2_train = b_model.score(X_train, y_train)  # Розрахунок коефіцієнта детермінації R^2
    r2_test = b_model.score(X_test, y_test)
    print('R^2 Score test:', r2_test)
    print('R^2 Score train:', r2_train)
    # Вивід коефіцієнтів
    print('Model coefficients:', b_model.coef_)
    print('Model intercept:', b_model.intercept_)
    print('\n\n')

    print("Градієнтний спуск:")
    solution(X_train, y_train, X_test, y_test, None)

    print("Градієнтний спуск з L1:")
    solution(X_train, y_train, X_test, y_test, ind="l1")

    print("Градієнтний спуск з L2:")
    solution(X_train, y_train, X_test, y_test, ind="l2")


if __name__ == '__main__':
    main()
