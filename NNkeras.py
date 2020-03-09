# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:57:06 2020

@author: Lenovo
"""



from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import tensorflow.keras as k

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

df=pd.read_csv('student-mat.csv',delimiter=';') 


school_map1 = {'GP': 1, 'MS': -1}
sex_map2 = {'M': 1, 'F': -1}
address_map4 = {'U': 1, 'R': -1}
famsize_map5 = {'LE3': 1, 'GT3': -1}
Pstatus_map6 = {'T': 1, 'A': -1}
schoolsup_map16 = {'yes': 1, 'no': -1}
famsup_map17 = {'yes': 1, 'no': -1}
paid_map18 = {'yes': 1, 'no': -1}
activities_map19 = {'yes': 1, 'no': -1}
nursery_map20 = {'yes': 1, 'no': -1}
higher_map21 = {'yes': 1, 'no': -1}
internet_map22 = {'yes': 1, 'no': -1}
romantic_map23 = {'yes': 1, 'no': -1}

df['school'] = df['school'].map(school_map1)
df['sex'] = df['sex'].map(sex_map2)
df['address'] = df['address'].map(address_map4)
df['famsize'] = df['famsize'].map(famsize_map5)
df['Pstatus'] = df['Pstatus'].map(Pstatus_map6)
df['schoolsup'] = df['schoolsup'].map(schoolsup_map16)
df['famsup'] = df['famsup'].map(famsup_map17)
df['paid'] = df['paid'].map(paid_map18)
df['activities'] = df['activities'].map(activities_map19)
df['nursery'] = df['nursery'].map(nursery_map20)
df['higher'] = df['higher'].map(higher_map21)
df['internet'] = df['internet'].map(internet_map22)
df['romantic'] = df['romantic'].map(romantic_map23)

G3=df['G3']
df = df.drop(columns="G3")

for i in range(len(G3)):
    if G3[i]<=4:
        G3[i]=0
    if G3[i]>=5 and G3[i]<=8:
        G3[i]=1
    if G3[i]>=9 and G3[i]<=12:
        G3[i]=2
    if G3[i]>=13 and G3[i]<=16:
        G3[i]=3
    if G3[i]>=17 and G3[i]<=20:
        G3[i]=4

df = pd.concat([df,G3],axis=1)

column_trans=make_column_transformer((OneHotEncoder(),['Mjob','Fjob',
'reason','guardian']),remainder='passthrough')

data=column_trans.fit_transform(df)
n1 = data.shape[0]
n2 = data.shape[1]
m=int(0.8*n1)
train=data[:m,:]
test=data[m:-1,:]

X_train=train[:,0:n2-1]
y_train=train[:,n2-1]
y_train = np.reshape(y_train, (len(y_train),1))
y_train = to_categorical(y_train)


X_test=test[:,0:n2-1]
y_test=test[:,n2-1]
y_test = np.reshape(y_test, (len(y_test),1))
y_test = to_categorical(y_test)


model = k.models.Sequential()
model.add(Dense(32, input_shape=(n2-1,)))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(5))


model.compile(optimizer='Adamax',
             loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train,
            epochs=500,
            shuffle=True)



score, acc = model.evaluate(X_test, y_test)
