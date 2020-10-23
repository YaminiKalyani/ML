"""
Last modified Oct 2020
@author: Miguel V. Martin for ML course
"""

import warnings
warnings.filterwarnings('ignore')

import urllib.request

#fetch data
target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data = urllib.request.urlopen(target_url)
xList = []
labels = []
names = []
firstLine = False # Turn to True if the data includes column names
for line in data:
    if firstLine:
        names = line.decode().strip().split(",")
        firstLine = False
    else:
        #split on coma
        row = line.decode().strip().split(",")
        #put labels in separate array
        labels.append(float(row[0]))
        #convert row to floats
        floatRow = [float(num) for num in row]
        xList.append(floatRow[1:]) #exclude first column, which is the label
 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
xList = scaler.fit_transform(xList)
    
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(9,5,3))#, random_state=42)
#clf.fit(xList, labels)  
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, xList, labels, cv=10, scoring="accuracy")
print('Score per fold:', scores)
    
