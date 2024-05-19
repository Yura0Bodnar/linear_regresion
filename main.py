import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def mserror(X, w, y):
    y_pred = X.dot(w)
    return np.sum((y - y_pred) ** 2) / len(y_pred)


def gradient_mse(X, w, y, lambda_l1):
    y_pred = X.dot(w)
    grad = 2/len(X) * X.T.dot(y_pred - y)
    # Додавання градієнта штрафу L1
    #l1_grad = lambda_l1 * np.sign(w)
    return grad #+ l1_grad


def main():
    data = pd.read_csv('adm_data.csv')
    X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
    y = data['Chance of Admit '].values
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Вбудована модель
    b_model = LinearRegression()
    b_model.fit(X, y)
    y_pred_b = b_model.predict(X)
    mse_b = mean_squared_error(y, y_pred_b)
    print('Sklearn MSE:', mse_b)

    # lambda_l1 = 0.01  # Регуляризаційний параметр L1
    w = np.zeros(X.shape[1])
    learning_rate = 0.01
    n_iterations = 2500
    eps = 1e-6
    mse = 0

    for i in range(n_iterations):
        grad = gradient_mse(X, w, y, None)
        new_w = w - learning_rate * grad
        mse = mserror(X, new_w, y)

        if i % 100 == 0:
            print(f"Iteration {i}: MSE = {mse}")

        if np.linalg.norm(new_w - w, ord=2) < eps:
            print('Convergence reached after', i+1, 'iterations.')
            break
        w = new_w

    print('Custom model coefficients:', w)
    print('Custom MSE:', mse)


if __name__ == '__main__':
    main()
