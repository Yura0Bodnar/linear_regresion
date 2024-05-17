import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv('adm_data.csv')

# Попередня обробка
# Видалення рядків, де відсутні дані по GRE Score або TOEFL Score
data = data.dropna(subset=['GRE Score', 'TOEFL Score'])

# Отримання векторів для незалежних та залежної змінних
X = data[['GRE Score']]  # Незалежна змінна
y = data['TOEFL Score']  # Залежна змінна

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення моделі лінійної регресії
model = LinearRegression()

# Тренування моделі
model.fit(X_train, y_train)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
res = model.score(X_train, y_train)

print('Mean Squared Error:', mse)
print("Efficiency: ", res)

# Коефіцієнти моделі
print('Slope:', model.coef_[0])
print('Intercept:', model.intercept_)

# Візуалізація результатів
plt.scatter(X_test, y_test, color='black')  # Справжні значення для тестової вибірки
plt.plot(X_test, y_pred, color='blue', linewidth=3)  # Лінія прогнозу
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.title('Linear Regression Analysis')
plt.show()
