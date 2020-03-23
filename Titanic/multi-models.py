# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:58:07 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import sklearn
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis
from sklearn import model_selection
from xgboost import XGBClassifier

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
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from collections import Counter
import math
from sklearn.impute import SimpleImputer 

df=pd.read_csv('train.csv',delimiter=',') 

#Embarked_map = {'S':0, 'C': 1, 'Q': 2}
sex_map = {'male': 1, 'female': 0}


#df['Embarked'] = df['Embarked'].map(Embarked_map)
df['Sex'] = df['Sex'].map(sex_map)


df = df.drop(columns="PassengerId")
df = df.drop(columns="Name")
df = df.drop(columns="Ticket")

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
df['Cabin'] = df['Cabin'].replace(np.nan, 'Unknown', regex=True)
for i in range(df.shape[0]):
    letter = df.loc[i,'Cabin']
    df.loc[i,'Cabin'] = letter[0]
df['Embarked'] = df['Embarked'].replace(np.nan, 'U', regex=True)

df = pd.concat([df,pd.get_dummies(df['Cabin'], prefix='Cabin')],axis=1)
df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1)

#column_trans=make_column_transformer((OneHotEncoder(),['Cabin']),remainder='passthrough')
df = df.drop(columns=["Cabin","Embarked"])
df = df.drop(columns="Cabin_T")
    


data=df.to_numpy()

#nan=[]
#n1 = data.shape[0]
#n2 = data.shape[1]
#for i in range(n1):
#    for j in range(n2):
#        if np.isnan(data[i,j]):
#            nan.append([i,j])
#m=int(0.8*n1)

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

#train=data[:m,:]
# train[:,0:(n2-1)] = preprocessing.scale(train[:,0:(n2-1)])
# scaler = preprocessing.StandardScaler().fit(train[:,0:(n2-1)])
# scaler = MinMaxScaler()
# scaler.fit(train[:,0:(n2-1)])


# train[:,0:(n2-1)]=scaler.transform(train[:,0:(n2-1)])
#test=data[m:-1,:]
# test[:,0:(n2-1)]=scaler.transform(test[:,0:(n2-1)])

train,test = train_test_split(df,train_size=0.8)
train_index = []
for r in train.index: 
    train_index.append(r)
test_index = []
for r in test.index: 
    test_index.append(r)
split = [(train_index, test_index)]



def sigmoid(z):
    sigmoid_f = 1 / (1 + np.exp(-z)) #YOUR CODE HERE
    return sigmoid_f 


# construct the data matrix X

#X = np.ones([n1,n2]) 
#X[:,1:n2] = data[:,0:(n2-1)]
#print(X.shape)
# print(X[:5,:])



# predictive function definition
#def f_pred(X,w): 
#    p = sigmoid(X.dot(w)) #YOUR CODE HERE
#    return p


# loss function definition
#def loss_logreg(y_pred,y): 
#    n = len(y)
#    loss = -1/n* ( y.T.dot(np.log(y_pred)) + (1-y).T.dot(np.log(1-y_pred)) ) #YOUR CODE HERE
#    return loss

# run logistic regression with scikit-learn
#start = time.time()
#logreg_sklearn = LogisticRegression(C=1e5,solver='sag',
#                                random_state=42) # scikit-learn logistic regression
#clf=logreg_sklearn.fit(train[:,0:(n2-1)], train[:,(n2-1)]) # learn the model parameters #YOUR CODE HERE
#print('Time=',time.time() - start)

Methods = [
    #Ensemble
    ensemble.AdaBoostClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    #Linear Model
    linear_model.LogisticRegression(max_iter = 5000),
    
    #Navies Bayes
#    naive_bayes.BernoulliNB(), # binary features
#    naive_bayes.GaussianNB(), # continuous features
    
    #K Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(max_iter = 5000,probability=True),
    svm.LinearSVC(max_iter = 5000),
    
    #Tree    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #XGBoost
    XGBClassifier()    
    ]

#create table
compare_columns = ['MLA Name','Test Accuracy Mean','Time']
MLA_compare = pd.DataFrame(columns = compare_columns)

#create table to compare MLA predictions
MLA_predict = []

#index through MLA and save performance to table
i = 0
x = df.iloc[:,1:-1]
y = df.iloc[:,0]
for alg in Methods:

    #set name and parameters
    name = alg.__class__.__name__
    MLA_compare.loc[i, 'MLA Name'] = name
    
    cv_results = model_selection.cross_validate(alg, x, y, cv = split)

    MLA_compare.loc[i, 'Time'] = cv_results['fit_time'].mean()
    #MLA_compare.loc[i, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[i, 'Test Accuracy Mean'] = cv_results['test_score'].mean()   
    
    alg.fit(x, y)
 #   MLA_predict[name] = alg.predict(x)
    
    i+=1

pd.set_option('display.max_columns', None)
print(MLA_compare)

# compute loss value
# w_sklearn = np.zeros([46,1])
# w_sklearn[0,0] = logreg_sklearn.intercept_
# w_sklearn[1:46,0] = logreg_sklearn.coef_
# print(w_sklearn)
# loss_sklearn = loss_logreg(f_pred(X,w_sklearn),data[:,(n2-1)][:,None])
# print('loss sklearn=',loss_sklearn)


#predict=clf.predict(test[:,0:(n2-1)])
#predict_proba=clf.predict_proba(test[:,0:(n2-1)])
#print('logistic regression')
#score=clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
# cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
#print('score',score)
# print('cross_validate_score',score)

#names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]



#classifiers = [
#    KNeighborsClassifier(3),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=2, C=1),
#    GaussianProcessClassifier(1.0 * RBF(1.0)),
#    DecisionTreeClassifier(max_depth=5),
#    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    MLPClassifier(alpha=1, max_iter=1000),
#    AdaBoostClassifier(),
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis()]



#s=[]

# iterate over classifiers
#for name, clf in zip(names, classifiers):
#    print(name)
#    clf.fit(train[:,0:(n2-1)], train[:,(n2-1)])
#    score = clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
    # cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
#    print('score',score)
    # print('cross_validate_score',score)
#    s.append(score)



test_df = pd.read_csv('test.csv',delimiter=',') 


#df['Embarked'] = df['Embarked'].map(Embarked_map)
test_df['Sex'] = test_df['Sex'].map(sex_map)

test_df = test_df.drop(columns="Name")
test_df = test_df.drop(columns="Ticket")

test_df[['Age']] = imputer.fit_transform(test_df[['Age']])
test_df['Cabin'] = test_df['Cabin'].replace(np.nan, 'Unknown', regex=True)
for i in range(test_df.shape[0]):
    letter = test_df.loc[i,'Cabin']
    test_df.loc[i,'Cabin'] = letter[0]
test_df['Embarked'] = test_df['Embarked'].replace(np.nan, 'U', regex=True)

test_df = pd.concat([test_df,pd.get_dummies(test_df['Cabin'], prefix='Cabin')],axis=1)
test_df = pd.concat([test_df,pd.get_dummies(test_df['Embarked'], prefix='Embarked')],axis=1)

#column_trans=make_column_transformer((OneHotEncoder(),['Cabin']),remainder='passthrough')
PassengerId = test_df.iloc[:,0]
test_df = test_df.drop(columns=["Cabin","Embarked","PassengerId"])

test_data=test_df.to_numpy()
test_data=test_data[:,1:-1]

xgb = XGBClassifier().fit(x, y)
predictions = xgb.predict(test_df)

Submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
Submission.to_csv("Submission.csv", index=False)