# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:58:07 2020

@author: Lenovo
"""


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

from sklearn.model_selection import GridSearchCV 
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
# train[:,0:(n2-1)] = preprocessing.scale(train[:,0:(n2-1)])
# scaler = preprocessing.StandardScaler().fit(train[:,0:(n2-1)])
# scaler = MinMaxScaler()
# scaler.fit(train[:,0:(n2-1)])


# train[:,0:(n2-1)]=scaler.transform(train[:,0:(n2-1)])
test=data[m:-1,:]
# test[:,0:(n2-1)]=scaler.transform(test[:,0:(n2-1)])

import xgboost as xgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
            
# data_dmatrix = xgb.core.DMatrix(data=data[:,0:(n2-1)],label=data[:,(n2-1)])

# params = {"objective":"multi:softmax",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10, 'num_class':2}

# cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

# xg_reg = xgb.XGBRegressor(objective ='multi:softmax', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10,num_class=2)

# xg_reg.fit(train[:,0:(n2-1)],train[:,(n2-1)])

# preds = xg_reg.predict(test[:,0:(n2-1)])
# accuracy_score(test[:,(n2-1)], preds)

# fit model no training data
# model = XGBClassifier()
# model.fit(train[:,0:(n2-1)],train[:,(n2-1)])
# 	
# print(model)

# preds = model.predict(test[:,0:(n2-1)])
# accuracy=accuracy_score(test[:,(n2-1)], preds)


xtrain=train[:,0:(n2-1)]
# xtrain = np.delete(xtrain, 5, 1)
# xtrain = np.delete(xtrain, 2, 1)
ytrain=train[:,(n2-1)]
xtest=test[:,0:(n2-1)]
ytest=test[:,(n2-1)]

# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot


# fit model no training data
model = XGBClassifier()
model.fit(xtrain, ytrain)
# plot feature importance
plot_importance(model)
pyplot.show()