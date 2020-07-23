# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:18:23 2020

@author: Lenovo
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:57:06 2020

@author: Lenovo
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras as k
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time
from keras.utils import to_categorical

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time

from sklearn.impute import SimpleImputer 
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
# train=data[:m,:]
# test=data[m:-1,:]

# X_train=train[:,0:n2-1]
# y_train=train[:,n2-1]
# y_train = np.reshape(y_train, (len(y_train),1))
# y_train = to_categorical(y_train)


# X_test=test[:,0:n2-1]
# y_test=test[:,n2-1]
# y_test = np.reshape(y_test, (len(y_test),1))
# y_test = to_categorical(y_test)

X = data[:,0:n2-1]
Y = data[:,n2-1]
Y = np.reshape(Y, (len(Y),1))


# define 10-fold cross validation test harness
seed = 7
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []






# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

sp=kfold.split(X, Y)
Y = to_categorical(Y)
print("Y: ", Y.shape)
test_accuracy=[]
for train, test in sp:
    model = k.models.Sequential()
    model.add(Dense(32, input_shape=(n2-1,),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='Adamax',
             loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history=model.fit(X[train], Y[train],validation_data=(X[test], Y[test]),
                epochs=1000,
                shuffle=True,verbose=1,callbacks=[es])
    # evaluate the model
    _, train_acc = model.evaluate(X[train], Y[train], verbose=1)
    _, test_acc = model.evaluate(X[test], Y[test], verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    
    test_accuracy.append(max(history.history['val_acc']))
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()



# score, acc = model.evaluate(X_test, y_test)
