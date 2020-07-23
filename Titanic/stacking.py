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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
df=pd.read_csv('train.csv',delimiter=',') 

count_no=[]
from collections import Counter
count = Counter(df['Name'])
for i in count:
    count_no.append(count[i])
    
count2 = Counter(df['Embarked'])

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
df['Cabin'] = df['Cabin'].replace(np.nan, 'a', regex=True)
df['Embarked'] = df['Embarked'].replace(np.nan, 3, regex=True)



for i in range(df.shape[0]):
    df['Cabin'][i]= df['Cabin'][i][0]
df = pd.concat([df,pd.get_dummies(df['Cabin'], prefix='Cabin')],axis=1)
# df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1)

# column_trans=make_column_transformer((OneHotEncoder(),['Cabin']),remainder='passthrough')
df = df.drop(columns="Cabin")
# df = df.drop(columns="Embarked")
df = df.drop(columns="Cabin_T")

Survived = df["Survived"]
df = df.drop(columns="Survived")
df = pd.concat([df,Survived],axis=1)

data=df.to_numpy()

nan=[]
n1 = data.shape[0]
n2 = data.shape[1]

# for i in range(n1):
#     for j in range(n2):
#         if np.isnan(data[i,j]):
#             nan.append([i,j])
m=int(0.8*n1)



#z-scoring
for i in range(n2-1):
    data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()  # Feature scaling
   
for i in range(n1):
        data[i,2]=data[i,2]/abs(max(data[:,2]))
        data[i,5]=data[i,5]/abs(max(data[:,5]))
# data[:,2] = (data[:,2] - data[:,2].mean())/data[:,2].std() 
# data[:,5] = (data[:,5] - data[:,5].mean())/data[:,5].std() 
# scaler = StandardScaler()
# data=scaler.fit(data) 

# for j in range(n2-1):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
#max normalization (only needed for some features since many of them are between 0 and 1 already)
# j=2
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
# j=5
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
    
# for j in range(n2-1):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
        
        
# data[:,5]=data[:,5]*10        
# data[:,5]=data[:,2]*10
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




# def sigmoid(z):
#     sigmoid_f = 1 / (1 + np.exp(-z)) #YOUR CODE HERE
#     return sigmoid_f 


# # construct the data matrix X

# X = np.ones([n1,n2]) 
# X[:,1:n2] = data[:,0:(n2-1)]
# print(X.shape)
# # print(X[:5,:])



# # predictive function definition
# def f_pred(X,w): 
#     p = sigmoid(X.dot(w)) #YOUR CODE HERE
#     return p


# # loss function definition
# def loss_logreg(y_pred,y): 
#     n = len(y)
#     loss = -1/n* ( y.T.dot(np.log(y_pred)) + (1-y).T.dot(np.log(1-y_pred)) ) #YOUR CODE HERE
#     return loss

# # run logistic regression with scikit-learn
# start = time.time()
# logreg_sklearn = LogisticRegression(C=1e5,solver='sag',
#                                 random_state=42) # scikit-learn logistic regression
# clf=logreg_sklearn.fit(train[:,0:(n2-1)], train[:,(n2-1)]) # learn the model parameters #YOUR CODE HERE
# print('Time=',time.time() - start)


# compute loss value
# w_sklearn = np.zeros([46,1])
# w_sklearn[0,0] = logreg_sklearn.intercept_
# w_sklearn[1:46,0] = logreg_sklearn.coef_
# print(w_sklearn)
# loss_sklearn = loss_logreg(f_pred(X,w_sklearn),data[:,(n2-1)][:,None])
# print('loss sklearn=',loss_sklearn)


# predict=clf.predict(test[:,0:(n2-1)])
# predict_proba=clf.predict_proba(test[:,0:(n2-1)])
# print('logistic regression')
# score=clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
# # cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
# print('score',score)
# print('cross_validate_score',score)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
          "Naive Bayes", "QDA","GradientBoosting","HistGradientBoosting","LogisticRegression"]



classifiers = [
    KNeighborsClassifier(n_neighbors=7,weights='distance'),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.01, early_stopping=False, max_iter=2000, solver='adam'),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),
    HistGradientBoostingClassifier(max_iter=100),
    LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000)]



s=[]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name)
    clf.fit(train[:,0:(n2-1)], train[:,(n2-1)])
    score = clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
    # cross_validate_score = cross_validate(clf, train[:,0:(n2-1)], train[:,(n2-1)], cv=3, scoring="roc_auc")["test_score"].mean()
    print('score',score)
    # print('cross_validate_score',score)
    s.append(score)



#bagging
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=10, random_state=0, warm_start=True)
bagging.fit(train[:,0:(n2-1)], train[:,(n2-1)])
bagging_acc=bagging.score(test[:,0:(n2-1)], test[:,(n2-1)])

#stacking
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
n3=test.shape[0]
# base_learners  = [
#     ('1_1',SVC(gamma=2, C=1,probability=True)),
#     ('1_2',DecisionTreeClassifier(max_depth=7)),
#     ('1_3',MLPClassifier(alpha=0.01, early_stopping=False, max_iter= 1000, solver='adam')),
#     ('1_4', KNeighborsClassifier(n_neighbors=7,weights='distance')),
#     ('1_5', BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=0, warm_start=True)),
#     ('1_6',GaussianProcessClassifier(0.949**2 * RBF(length_scale=1))),
#     ('1_7',GradientBoostingClassifier(n_estimators=100, learning_rate= 0.1, loss= 'exponential', max_depth= 4, warm_start= True, random_state=0)),
#     # ('1_8',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
#     # ('1_9',HistGradientBoostingClassifier(max_iter=100))
    
#     ]

base_learners  = [('1_1',SVC(gamma=2, C=1,probability=True)),
                  ('1_2',DecisionTreeClassifier(max_depth=7)),
                  ('1_3',MLPClassifier(alpha=1, max_iter=1000)),
                  ('1_4', KNeighborsClassifier(n_neighbors=20,weights='uniform')),
                  ('1_5', BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10, random_state=0, warm_start=True)),
                   ('1_6',GaussianProcessClassifier(0.949**2 * RBF(length_scale=1))),
                   ('1_7',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)) 
                  ]
layer_two_estimators = [
                        ('dt_2', DecisionTreeClassifier()),
                        ('rf_2', RandomForestClassifier(n_estimators=50, random_state=42)),
                       ]
layer_two = StackingClassifier(estimators=layer_two_estimators, final_estimator=LogisticRegression())
# clf = StackingClassifier(estimators=base_learners, final_estimator=RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
stack_clf = StackingClassifier(estimators=base_learners,
                          final_estimator=LogisticRegression(),  
                          cv=10)
stack_clf.fit(train[:,0:(n2-1)], train[:,(n2-1)])
stack_acc=stack_clf.score(test[:,0:(n2-1)], test[:,(n2-1)])
    
#voting
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

v_clf = VotingClassifier(estimators=base_learners,voting='soft')
v_clf.fit(data[:,0:(n2-1)], data[:,(n2-1)])
v_scores = cross_val_score(v_clf, data[:,0:(n2-1)], data[:,(n2-1)], scoring='accuracy', cv=5).mean()
    
    
df=pd.read_csv('test.csv',delimiter=',') 


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
df[['Fare']] = imputer.fit_transform(df[['Fare']])


#data cleaning (replace NaN in categorical features with the same string can try other methods)
df['Cabin'] = df['Cabin'].replace(np.nan, 'aa', regex=True)
df['Embarked'] = df['Embarked'].replace(np.nan, 3, regex=True)
for i in range(df.shape[0]):
    df['Cabin'][i]= df['Cabin'][i][0]
df = pd.concat([df,pd.get_dummies(df['Cabin'], prefix='Cabin')],axis=1)
# df = pd.concat([df,pd.get_dummies(df['Embarked'], prefix='Embarked')],axis=1)
# column_trans=make_column_transformer((OneHotEncoder(),['Cabin']),remainder='passthrough')
df = df.drop(columns="Cabin")
# df = df.drop(columns="Embarked")

data=df.to_numpy()
# b = np.zeros([data.shape[0], 1])
# data=np.concatenate((data, b), axis=1)
nan=[]
n1 = data.shape[0]
n2 = data.shape[1]
for i in range(n1):
    for j in range(n2):
        if np.isnan(data[i,j]):
            nan.append([i,j])
m=int(0.8*n1)


# z-scoring
for i in range(n2-1):
    data[:,i] = (data[:,i] - data[:,i].mean())/data[:,i].std()  # Feature scaling
# data[:,2] = (data[:,2] - data[:,2].mean())/data[:,2].std() 
# data[:,5] = (data[:,5] - data[:,5].mean())/data[:,5].std() 
# scaler = StandardScaler()
# data=scaler.fit(data) 
for i in range(n1):
        data[i,2]=data[i,2]/abs(max(data[:,2]))
        data[i,5]=data[i,5]/abs(max(data[:,5]))
# for j in range(n2-1):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
#max normalization (only needed for some features since many of them are between 0 and 1 already)
# j=2
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
# j=5
# for i in range(n1):
#     data[i,j]=data[i,j]/abs(max(data[:,j]))
    
    
# for j in range(6):    
#     for i in range(n1):
#         data[i,j]=data[i,j]/abs(max(data[:,j]))
n3=data.shape[0]
predicts=v_clf.predict(data[:,:])

    
df=pd.read_csv('submission.csv',delimiter=',') 

Survivedd={'Survived':predicts}

# for i in range(n3):
#     Survivedd[Survivedd]=predicts[i]
df2 = pd.DataFrame(Survivedd) 
df = pd.concat([df,df2],axis=1)

df.to_csv(r'voting_submission.csv', index = False)

predicts=stack_clf.predict(data[:,:])
df=pd.read_csv('submission.csv',delimiter=',') 
Survivedd={'Survived':predicts}
df2 = pd.DataFrame(Survivedd) 
df = pd.concat([df,df2],axis=1)

df.to_csv(r'stacking_submission.csv', index = False)