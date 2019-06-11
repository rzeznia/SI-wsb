# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


from sklearn.preprocessing import StandardScaler

# Utworzenie obiektów sc_X i sc_y.
sc_X = StandardScaler()
sc_y = StandardScaler()

# Dopasowanie za pomocą metody fit_transform
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


from sklearn.svm import SVR

# Tworzenie obiektu klasy SVR z argumentem kernel='rbf', czyli ze wskazaniem że funkcja jądra 'rbf'
regressor=SVR(kernel='rbf')

# Dopasowanie modelu
regressor.fit(X, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Wykreślenie danych z większą rozdzielczością
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.0]]))))


