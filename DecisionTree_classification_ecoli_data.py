# -*- coding: utf-8 -*-
# decision tree classification
# type: multiclass classification
# dataset: ecoli

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# draw the tree
from sklearn import tree
import matplotlib.pyplot as plt

# feature selection
# RFE (recursive feature elimination)
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# read the data
path="F:/aegis/4 ml/dataset/supervised/classification/ecoli/ecoli.csv" 
ecoli=pd.read_csv(path)

ecoli.head()
ecoli.shape
ecoli.dtypes

# remove 'sequence_name' from dataset
ecoli.drop(columns='sequence_name',inplace=True)
ecoli.columns
ecoli.head()

# get the count of classes
ecoli.lsp.value_counts()

# check for singularities on columns 'chg' and 'lip'
ecoli.lip.value_counts()
len(ecoli)
# 326/336

ecoli.chg.value_counts()
335/336

# columns 'chg' and 'lip' have singularities. (prop of same value more than 85%). remove these features
# drop these after the first model
ecoli.drop(columns=['chg','lip'],inplace=True)
ecoli.head()

# shuffle the dataset
ecoli = ecoli.sample(frac=1)
ecoli.head(20)

# split the dataset
trainx1,testx1,trainy1,testy1 = train_test_split(ecoli.drop('lsp',1),
                                                 ecoli.lsp,
                                                 test_size=0.25)

trainx1.shape,trainy1.shape
testx1.shape,testy1.shape

# model building 
# i) entropy model
# ii) gini index model
    # without hyperparameters (default values)
    # with hyperparameter tuning (gridsearch CV)/ randomsearch CV)

'''
imp hyper parameters:
1) criterion: gini, entropy
2) max_depth: control the depth of the tree
3) min_samples_split: how many minimum samples in each splot
4) min_samples_leaf: mis samples needed in the leaf node
5) max_features: number of features for splitting
'''

# Grid Search

from sklearn.model_selection import GridSearchCV

# create an instance of the DecisionTreeClassifier class
dtclf = DecisionTreeClassifier()

# build the Hyperparameters
params = {"criterion":["gini","entropy"],
          "max_depth": np.arange(3,8),
          "min_samples_leaf":np.arange(2,11),
          "min_samples_split":np.arange(2,11) }

# run the GridSearch CV
grid = GridSearchCV(dtclf,param_grid=params,scoring='accuracy',
                    cv=10,n_jobs=-1).fit(trainx1,trainy1)

# check for the combination of parameters that gives the best accuracy
grid.best_params_
grid.best_score_

# build the model using the GridSearch results for best params
m1 = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=2,
                            min_samples_split=2).fit(trainx1,trainy1)

# predictions on test data
p1 = m1.predict(testx1)

# confusion matrix and classification report
df1 = pd.DataFrame({'actual':testy1,'predicted':p1})
pd.crosstab(df1.actual,df1.predicted,margins=True)
print(classification_report(df1.actual,df1.predicted))

# Feature Selection
# method 1 : from the model
impf = pd.DataFrame({'feature':trainx1.columns,
                     'score': m1.feature_importances_})

impf = impf.sort_values('score',ascending=False)
print(impf)


# method 2: recursive feature elimination
top = 5
rfe = RFE(m1,n_features_to_select = top).fit(trainx1,trainy1)
rfe.support_
rfe.ranking_

impf=pd.DataFrame({'feature':trainx1.columns,
                   'support':rfe.support_,
                   'rank':rfe.ranking_})

impf.sort_values('rank')


# remove the features 'lip' and 'chg' from the dataset and build model 2
# compare model1 and model2


# pruning the Decision Tree using the Cost Complexity parameter

path = m1.cost_complexity_pruning_path(trainx1,trainy1)
print(path)

alphas = path.ccp_alphas

# loop through every alpha value, build model and determine the best alpha

clfs = [] # store the list of models for each alpha value

for a in alphas:
    model = DecisionTreeClassifier(ccp_alpha=a).fit(trainx1,trainy1)
    clfs.append(model)

print(clfs)


# compute the train and test score for each model
trainscore = [clf.score(trainx1,trainy1)  for clf in clfs]
testscore = [clf.score(testx1,testy1) for clf in clfs]

# plot the train and test score to determine the best alpha
fig,ax = plt.subplots()
ax.plot(alphas,trainscore,marker='x',label="train",drawstyle="steps-post")
ax.plot(alphas,testscore,marker='x',label="test",drawstyle="steps-post")
ax.set_xlabel("Alphas")
ax.set_ylabel("Accuracy")
ax.set_title("Alphas vc Accuracy - CCP")
ax.legend()

# according to the above model, the best cp value is 0.0106
# build next model using this Cost Complexity Parameter value
ccp = 0.0106
m2 = DecisionTreeClassifier(criterion='gini',
                            max_depth=4, 
                            min_samples_leaf=2,
                            min_samples_split=2,
                            ccp_alpha=ccp).fit(trainx1,trainy1)
 
p2 = m2.predict(testx1)

# confusion matrix and classification report
df2 = pd.DataFrame({'actual':testy1,'predicted':p2})
pd.crosstab(df2.actual,df2.predicted,margins=True)
print(classification_report(df2.actual,df2.predicted))

# plot the decision tree
fig = plt.figure(figsize=(15,10))
tree.plot_tree(m1,
               feature_names=trainx1.columns,
               class_names=trainy1.unique(),
               filled=True)