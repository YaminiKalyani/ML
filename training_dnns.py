# -*- coding: utf-8 -*-
"""training_DNNs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jyH8eAobbttfEsf6AAl5FU_q95K9uW1d

Last modified: Oct 2020

Author: Miguel V. Martin for ML course, based on Geron

We will use our usual attributes_vs_salary dataset. This is a very shallow dataset but we only use it to illustrate concepts
"""

import pandas as pd
url = 'https://raw.githubusercontent.com/miguevmartin/INFR3700/master/attributes_vs_salary.dat'
sal_data = pd.read_csv(url)
#sal_data["Hobby_cat"] = sal_data["Hobby"].astype('category')
#sal_data.drop('Hobby') OR:
sal_data['Hobby'].value_counts()

"""We will predict the hobby, which is categorical, se we need to convert it to numerical values"""

hobby_codes = {'Hobby': {"running": 0, "cycling": 1, "handcraft": 2, "tv": 3, "reading": 4, 
                         "writing": 5, "cooking": 6, "biking": 7, "swimming": 8, "soccer": 9}}
sal_data.replace(hobby_codes, inplace=True)
sal_data.head()

"""Next, we apply our usual split but this time we do a 50/50 split to match the size of the training set with the validation set"""

from sklearn.model_selection import train_test_split
train, test = train_test_split(sal_data, test_size=0.5)#, random_state=42)
train_labels = train.iloc[:,-1]
train_data = train.drop(['Person','Hobby'], axis=1)
test_labels = test.iloc[:,-1]
test_data = test.drop(['Person','Hobby'], axis=1)
# For multiclass classification, we will use 'Hobby' as the target
train_labels_mc = train['Hobby']
test_labels_mc = test['Hobby']

"""Scale as usual:"""

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(train_data)

"""Let's see the versions available in Colab"""

import tensorflow as tf
from tensorflow import keras
print('tf version:',tf.__version__, 'keras version:',keras.__version__)

"""Available activations in Keras:"""

#[name for name in dir(keras.activations) if not name.startswith("_")]

"""Then we create a fully connected neural network"""

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[5]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, activation="relu", kernel_initializer="HeNormal"),
keras.layers.Dense(100, activation="elu"),
keras.layers.Dropout(rate=0.5),
keras.layers.Dense(10, activation="softmax")
])

"""Available initializers in Keras:"""

#[name for name in dir(keras.initializers) if not name.startswith("_")]

"""Visualize the model:"""

model.summary()

keras.utils.plot_model(model, "attributes_vs_salary_model.png", show_shapes=True) # Try rankdir = 'LR'

"""And then establish loss, optimizer, and metric, but modify parameters for better performance of deep nets"""

import keras.optimizers
#opt = keras.optimizers.RMSprop(learning_rate=0.01,
#    rho=0.09,
#    momentum=0.1,
#    epsilon=1e-07)
opt = keras.optimizers.Nadam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss="sparse_categorical_crossentropy",
optimizer=opt,
metrics=["accuracy"])

"""We will use the history object to create a graph"""

history = model.fit(train_data, train_labels_mc, epochs=10, validation_data=(test_data, test_labels_mc), verbose=0)# Turn verbose=1 to printing epochs

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 10)
plt.show()