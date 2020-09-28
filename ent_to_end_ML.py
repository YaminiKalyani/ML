# -*- coding: utf-8 -*-
"""
Modified Sept 2020

@author: Miguel V. Martin for ML course, following Geron's
"""

import pandas as pd
sal_data = pd.read_csv('attributes_vs_salary.dat')

#print(sal_data)
#print(sal_data.head(3))
#print(sal_data.info())
#print(sal_data["Hobby"].value_counts())
#print(sal_data.describe())

'''
import matplotlib.pyplot as plt
sal_data.hist(bins=50, figsize=(20,15))
plt.show()
'''

'''
import matplotlib.pyplot as plt
sal_data.plot(kind="scatter", x="Years of education", y="Income ($K/year)", alpha=0.4,
    s=sal_data['Age']*sal_data['Age'], label="Age", figsize=(10,7),
    c="GPA", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
plt.show()
'''

'''
sal_data['graduation_years'] = sal_data['Age'] - sal_data['Years of experience']
corr_matrix = sal_data.corr()
print(corr_matrix["Income ($K/year)"].sort_values(ascending=False))
'''

'''
sal_cat = sal_data['Hobby']
sal_cat_encoded, sal_categories = sal_cat.factorize()
print(sal_cat_encoded[:10])
print(sal_categories)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
print(sal_cat_encoded.reshape(-1,1))
sal_cat_1hot = encoder.fit_transform(sal_cat_encoded.reshape(-1,1))
sal_cat_1hot.toarray()
print(sal_cat_1hot.toarray())
'''


sal_data['Years of education'].hist()
import numpy as np
sal_data['educ_level'] = np.ceil(sal_data['Years of education'] / 6)
# Label those above 4 as 4
sal_data['educ_level'].where(sal_data['educ_level'] < 4, 4.0, inplace=True)
print(sal_data['educ_level'].value_counts())
#sal_data['educ_level'].hist()
#from sklearn.model_selection import StratifiedShuffleSplit
#split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
sal_data.drop('educ_level', axis=1, inplace=True)



from sklearn.model_selection import train_test_split
train, test = train_test_split(sal_data, test_size=0.2, random_state=42)
train_labels = train.iloc[:,-1]
train_data = train.drop(['Person','Income ($K/year)','Hobby'], axis=1)
test_labels = test.iloc[:,-1]
test_data = test.drop(['Person','Income ($K/year)','Hobby'], axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_data)
print(scaler.mean_)
train_data_tr = scaler.transform(train_data)
#scaler.fit(test_data) # Dont!
test_data_tr = scaler.transform(test_data)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_data_tr, train_labels)
print("Predictions:", lin_reg.predict(test_data_tr))

from sklearn.metrics import mean_squared_error
import numpy as np
sal_predictions = lin_reg.predict(test_data_tr)
lin_mse = mean_squared_error(test_labels, sal_predictions)
lin_rmse = np.sqrt(lin_mse)
print('linear regression rmse =', lin_rmse)
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(test_labels, sal_predictions)
print('linear regression mae =', lin_mae)


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_data_tr, train_labels)
sal_predictions = tree_reg.predict(test_data_tr)
tree_mse = mean_squared_error(test_labels, sal_predictions)
tree_rmse = np.sqrt(tree_mse)
print('decision tree rmse =', tree_rmse)



from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, train_data_tr, train_labels,
scoring="neg_mean_squared_error", cv=4)
tree_rmse_scores = np.sqrt(-scores)
print("scores:", tree_rmse_scores)
print("mean:", tree_rmse_scores.mean())
print("standard deviation:", tree_rmse_scores.std())
print(scores)

