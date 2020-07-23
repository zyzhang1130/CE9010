# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 01:41:04 2020

@author: Lenovo
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC




import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import time
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from collections import Counter
import math
from sklearn.impute import SimpleImputer 
from collections import Counter
import statistics
df=pd.read_csv('train.csv',delimiter=',') 


Embarked_map = {'S':0, 'C': 1, 'Q': 2}
sex_map = {'male': 1, 'female': 0}


df['Embarked'] = df['Embarked'].map(Embarked_map)
df['Sex'] = df['Sex'].map(sex_map)


df = df.drop(columns="PassengerId")
df = df.drop(columns="Name")
df = df.drop(columns="Ticket")

#data cleaning (replace this missing numerical feature entries with the median of that entry. Can try other methods)
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])


#data cleaning (replace NaN in categorical features with the same string can try other methods)
df['Cabin'] = df['Cabin'].replace(np.nan, 'aa', regex=True)
df['Embarked'] = df['Embarked'].replace(np.nan, 3, regex=True)
for i in range(df.shape[0]):
    df['Cabin'][i]= df['Cabin'][i][0]
df = pd.concat([df,pd.get_dummies(df['Cabin'], prefix='Cabin')],axis=1)

# column_trans=make_column_transformer((OneHotEncoder(),['Cabin']),remainder='passthrough')
df = df.drop(columns="Cabin")
df = df.drop(columns="Cabin_T")

Survived = df["Survived"]
df = df.drop(columns="Survived")
df = pd.concat([df,Survived],axis=1)

data=df.to_numpy()

nan=[]
n1 = data.shape[0]
n2 = data.shape[1]
for i in range(n1):
    for j in range(n2):
        if np.isnan(data[i,j]):
            nan.append([i,j])
m=int(0.8*n1)

#z-scoring
for i in range(n2-1):
    data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()  # Feature scaling

# scaler = StandardScaler()
# data=scaler.fit(data) 


#max normalization (only needed for some features since many of them are between 0 and 1 already)
# j=2
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
# j=5
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
    
for j in range(6):    
    for i in range(n1):
        data[i,j]=data[i,j]/abs(max(data[:,j]))
        
# for j in range(36,45):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
    

# data[:,0:(n2-1)] = data[:,0:(n2-1)]*100

train=data[:m,:]

X_train, y_train=train[:,0:(n2-1)], train[:,(n2-1)]
# train[:,0:(n2-1)] = preprocessing.scale(train[:,0:(n2-1)])
# scaler = preprocessing.StandardScaler().fit(train[:,0:(n2-1)])
# scaler = MinMaxScaler()
# scaler.fit(train[:,0:(n2-1)])


# train[:,0:(n2-1)]=scaler.transform(train[:,0:(n2-1)])
test=data[m:-1,:]

X_test, y_test=test[:,0:(n2-1)], test[:,(n2-1)]









# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf', 'poly','sigmoid'], 'gamma': [1e-1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1,2,3,4,5,6,7,8,9, 10],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()