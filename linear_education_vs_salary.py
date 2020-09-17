# -*- coding: utf-8 -*-
"""
Modified Sept 2020

@author: Miguel V. Martin for ML course
"""
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
plt.scatter(xList, labels, color = 'b')
plt.xlabel("years of education")
plt.ylabel("salary (in K$)")

# Two guesstimate models:
#plt.plot([0, 31],[45, 200], 'r--') #pred=45+5x; T0=45, T1=5 
#plt.plot([0, 31],[65, 130], 'g-.') #pred=65+2.1x; T0=45, T1=2.1; 

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
plt.plot(x_pred, y_pred, 'k', lw='5')

print('Theta0 = ', model.intercept_[0], ', Theta1 = ', model.coef_[0,0])

# Now suppose you have a new student for whom you want to predict salary
new_years_of_education = [[15]] 
print(model.predict(new_years_of_education))
plt.plot(new_years_of_education, model.predict(new_years_of_education), marker='s', color='red')

plt.show()

prediction = model.predict(X).flat
error = []
for i in range(len(labels)):
    error.append(labels[i] - prediction[i])

#print the errors
print("Errors ",)
print(error)

#calculate the squared errors and absolute value of errors
squaredError = []
absError = []
for val in error:
    squaredError.append(val*val)
    absError.append(abs(val))

#print squared errors and absolute value of errors
print("Squared Error")
print(squaredError)
print("Absolute Value of Error")
print(absError)

#calculate and print mean squared error MSE
print("MSE = ", sum(squaredError)/len(squaredError))

from math import sqrt
#calculate and print square root of MSE (RMSE)
print("RMSE = ", sqrt(sum(squaredError)/len(squaredError)))

#calculate and print mean absolute error MAE
print("MAE = ", sum(absError)/len(absError))

#compare MSE to target variance
targetDeviation = []
targetMean = sum(labels)/len(labels)
for val in labels:
    targetDeviation.append((val - targetMean)*(val - targetMean))

target_variance = sum(targetDeviation)/(len(targetDeviation)-1)
#print the target variance
print("Target Variance = ", target_variance)

target_std = sqrt(target_variance)
#print the the target standard deviation (square root of variance)
print("Target Standard Deviation = ", target_std)