# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:58:07 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.linear_model import LogisticRegression
import time
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
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv('student-mat.csv',delimiter=';') 


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
j=19
for i in range(n1):
    data[i,j]=data[i,j]/abs(max(data[:,j]))
    
    
for j in range(23,28):    
    for i in range(n1):
        data[i,j]=data[i,j]/abs(max(data[:,j]))
        
for j in range(36,45):    
    for i in range(n1):
        data[i,j]=data[i,j]/abs(max(data[:,j]))
    

data[:,0:(n2-1)] = data[:,0:(n2-1)]*100

train=data[:m,:]
# train[:,0:(n2-1)] = preprocessing.scale(train[:,0:(n2-1)])
# scaler = preprocessing.StandardScaler().fit(train[:,0:(n2-1)])
# scaler = MinMaxScaler()
# scaler.fit(train[:,0:(n2-1)])


# train[:,0:(n2-1)]=scaler.transform(train[:,0:(n2-1)])
test=data[m:-1,:]
# test[:,0:(n2-1)]=scaler.transform(test[:,0:(n2-1)])




def sigmoid(z):
    sigmoid_f = 1 / (1 + np.exp(-z)) #YOUR CODE HERE
    return sigmoid_f 


# construct the data matrix X

X = np.ones([n1,n2]) 
X[:,1:n2] = data[:,0:(n2-1)]
print(X.shape)
# print(X[:5,:])



# predictive function definition
def f_pred(X,w): 
    p = sigmoid(X.dot(w)) #YOUR CODE HERE
    return p


# loss function definition
def loss_logreg(y_pred,y): 
    n = len(y)
    loss = -1/n* ( y.T.dot(np.log(y_pred)) + (1-y).T.dot(np.log(1-y_pred)) ) #YOUR CODE HERE
    return loss

# run logistic regression with scikit-learn
start = time.time()
logreg_sklearn = LogisticRegression(C=1e5,solver='sag',
                                random_state=42) # scikit-learn logistic regression
clf=logreg_sklearn.fit(train[:,0:45], train[:,45]) # learn the model parameters #YOUR CODE HERE
print('Time=',time.time() - start)


# compute loss value
# w_sklearn = np.zeros([46,1])
# w_sklearn[0,0] = logreg_sklearn.intercept_
# w_sklearn[1:46,0] = logreg_sklearn.coef_
# print(w_sklearn)
# loss_sklearn = loss_logreg(f_pred(X,w_sklearn),data[:,(n2-1)][:,None])
# print('loss sklearn=',loss_sklearn)


predict=clf.predict(test[:,0:(n2-1)])
predict_proba=clf.predict_proba(test[:,0:(n2-1)])
print('logistic regression')
score=clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
# cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
print('score',score)
# print('cross_validate_score',score)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
          "Naive Bayes", "QDA"]



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]



s=[]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(train[:,0:45], train[:,45])
    score = clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
    # cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
    print('score',score)
    # print('cross_validate_score',score)
    s.append(score)
