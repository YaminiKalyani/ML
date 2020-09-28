# -*- coding: utf-8 -*-
"""
Modified Sept 2020

@author: Miguel V. Martin for the ML corse, adapted from Geron
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
data = open('attributes_vs_salary.dat', 'r')

xList = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.split(",")
        firstLine = False
    else:
        #split on comma
        row = line.split(",")
        #put labels in separate array
        labels.append(float(row[-1]))
        #convert row to floats
        xList.append(float(row[1]))

data.close()

# Plot points
plt.scatter(xList, labels, color = 'k')
plt.xlabel("years of education") 
plt.ylabel("salary (in K$)")

# Two guesstimate models:
plt.plot([0, 31],[45, 200], 'r:', label="Guesstimate1") #pred=45+5x; T0=45, T1=5 
plt.plot([0, 31],[65, 130], 'g-.', label='Guestimate2') #pred=65+2.1x; T0=45, T1=2.1; 

# A linear regression model finds the "best" fitting Thetas
model = linear_model.LinearRegression()

# Train linear model to obtain best Theta0 and Theta1
# But first turn lists into arrays
X = np.c_[xList]
y = np.c_[labels]
model.fit(X,y)

pred = model.predict([[31]])
x_pred = [0, 31]
y_pred = [model.intercept_[0], pred[0,0]]
plt.plot(x_pred, y_pred, 'k:', lw='5', label='SGD')
print('Theta0 = ', model.intercept_[0], ', Theta1 = ', model.coef_[0,0])


# Normal equation (no machine learning; just Linear Algebra)
X_b = np.c_[np.ones((len(X), 1)), xList] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(labels)
print('Normal equation:', theta_best)


# Batch GD can be easily implemented as follows; just beware of eta and iter values
eta = 0.001 # learning rate
iter = 10000
m = X.size
theta = np.random.randn(2,1) # random initialization
for iteration in range(iter):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
print("BatchGD Thetas:", theta)

# A linear model doesn't seem appropriate. Try Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False) #quadratic
X_poly = poly_features.fit_transform(X) # Square each feature and add as feature
print('Feature (before):', X[0], ', features (after):', X_poly[0])
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_poly, y)
print('lin_reg Thetas:', lin_reg.intercept_, lin_reg.coef_)
# Plot our new quadratic model
X_new=np.linspace(0, 31, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X_new, y_new, "m-", linewidth=2, label="PolynomialSGD")

# Now make predictions with each model and compare
pred = [[15]]
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty="l2", random_state=42, max_iter = 50, tol=1e-3) #sgd with l2 = Ridge
sgd_reg.fit(X, y)#.ravel())
y_new = sgd_reg.predict(X_new)
plt.plot(X_new, y_new, "b-", linewidth=2, label="RidgeSGD")
print("sgd:", sgd_reg.predict(pred))



from sklearn.linear_model import Ridge #try making alpha very big or very small and compare with Ridge equation
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42) #same as sgd with l2
ridge_reg.fit(X, y)
y_new = ridge_reg.predict(X_new)
plt.plot(X_new, y_new, "k:", linewidth=2, label="RidgeCholesky")
print('Cholesky:', ridge_reg.predict(pred))



ridge_reg = Ridge(alpha=1, solver="sag", random_state=42) #same as sgd with l2
ridge_reg.fit(X, y)
y_new = ridge_reg.predict(X_new)
plt.plot(X_new, y_new, "y+", linewidth=2, label="RidgeSGD")
print("sag:", ridge_reg.predict(pred))



from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1, random_state=42)
lasso_reg.fit(X, y)
y_new = ridge_reg.predict(X_new)
plt.plot(X_new, y_new, "c-", linewidth=2, label="Lasso")
print("lasso:", ridge_reg.predict(pred))


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
y_new = ridge_reg.predict(X_new)
plt.plot(X_new, y_new, "g-", linewidth=2, label="Elastic")
print("elastic:", elastic_net.predict(pred))


plt.legend()
plt.show()
