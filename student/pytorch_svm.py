# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:56:02 2020

@author: Lenovo
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:47:54 2020

@author: Lenovo
"""


# import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time

df=pd.read_csv('student-por.csv',delimiter=';') 


school_map1 = {'GP': 1, 'MS': 0}
sex_map2 = {'M': 1, 'F': 0}
address_map4 = {'U': 1, 'R': 0}
famsize_map5 = {'LE3': 1, 'GT3': 0}
Pstatus_map6 = {'T': 1, 'A': 0}
schoolsup_map16 = {'yes': 1, 'no': 0}
famsup_map17 = {'yes': 1, 'no': 0}
paid_map18 = {'yes': 1, 'no': 0}
activities_map19 = {'yes': 1, 'no': 0}
nursery_map20 = {'yes': 1, 'no': 0}
higher_map21 = {'yes': 1, 'no': 0}
internet_map22 = {'yes': 1, 'no': 0}
romantic_map23 = {'yes': 1, 'no': 0}

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

#z-scoring
# for i in range(n2-1):
#     data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()  # Feature scaling

# scaler = StandardScaler()
# data=scaler.fit(data) 


#max normalization (only needed for some features since many of them are between 0 and 1 already)
# j=19
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
    
# for j in range(23,28):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
        
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

X,Y=train[:,0:(n2-1)], train[:,(n2-1)]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

print(torch.cuda.is_available())

class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each 
    input sample is 2 and output sample  is 1.
    """
    def __init__(self):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(2, 1)  # Implement the Linear function
        
    def forward(self, x):
        fwd = self.fully_connected(x)  # Forward pass
        return fwd
    
    
data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
Y[Y == 0] = -1  # Replace zeros with -1
# plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
# plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling


learning_rate = 0.1  # Learning rate
epoch = 10  # Number of epochs
batch_size = 1  # Batch size

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)
N = len(Y)  # Number of samples, 500

model = SVM()  # Our model
# model.cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Our optimizer
model.train()  # Our model, SVM is a subclass of the nn.Module, so it inherits the train method
for epoch in range(epoch):
    perm = torch.randperm(N)  # Generate a set of random numbers of length: sample size
    sum_loss = 0  # Loss for each epoch
        
    for i in range(0, N, batch_size):
        x = X[perm[i:i + batch_size]]  # Pick random samples by iterating over random permutation
        y = Y[perm[i:i + batch_size]]  # Pick the correlating class
        
        x = Variable(x)  # Convert features and classes to variables
        y = Variable(y)

        optimizer.zero_grad()  # Manually zero the gradient buffers of the optimizer
        output = model(x)  # Compute the output by doing a forward pass
        
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize and adjust weights

        sum_loss += loss.item()  # Add the loss
        
    print("Epoch {}, Loss: {}".format(epoch, sum_loss))
    
X,Y=test[:,0:(n2-1)], test[:,(n2-1)]
data = X  # Before feature scaling
X = (X - X.mean())/X.std()  # Feature scaling
Y[Y == 0] = -1  # Replace zeros with -1
# plt.scatter(x=X[:, 0], y=X[:, 1])  # After feature scaling
# plt.scatter(x=data[:, 0], y=data[:, 1], c='r')  # Before feature scaling

X = torch.FloatTensor(X)  # Convert X and Y to FloatTensors
Y = torch.FloatTensor(Y)

x = Variable(x)  # Convert features and classes to variables
y = Variable(y)
output = model(x)
print(output) 
