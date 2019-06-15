# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Wysokość wynagrodzenia na poziomie stanowiska')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

##### DEGREE

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # <- DEGREE

X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color= 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

arr=np.array([6])
arr=arr.reshape(1,-1)
lin_reg.predict(arr)
lin_reg_2.predict(poly_reg.fit_transform(arr))




