# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Tworzenie obiektu klasy RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)

# Dopasowanie modelu
regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


arr=np.array([6])
arr=arr.reshape(1,-1)
y_pred = regressor.predict(arr)
