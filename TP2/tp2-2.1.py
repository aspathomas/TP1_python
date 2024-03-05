#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
rdforest = RandomForestClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, rdforest, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression', 'SVM']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        scores = cross_val_score(clf, X, y, cv=skf)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        skf = StratifiedKFold(n_splits=5,shuffle=True)
        predicted = cross_val_predict(clf, X, y, cv=skf) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))



marketing = pd.read_csv('data/marketing_campaign.csv', sep=',', na_values='')

# binary classes to categorical
bin_cols = [] # TO COMPLETE
marketing[bin_cols] = marketing[bin_cols].replace(0,'0').replace(1,'1')
print(marketing.info())



# Replace missing values by mean and scale numeric values
data_num = data.select_dtypes(include='number')





# Ddiscretize categorical values
data_cat = data.select_dtypes(exclude='number').drop('class',axis=1)


# Discretization with OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore")
# encoder.fit(X_cat)
# X_cat = encoder.transform(X_cat).toarray()


