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
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler

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
nombre_attributs = len(marketing.columns)
print(f"Nombre d'attributs: {nombre_attributs}")

# Sélectionner uniquement les colonnes numériques du DataFrame
attributs_numeriques = marketing.select_dtypes(include='number')

bin_cols = []
# Itérer sur chaque colonne numérique
for colonne in attributs_numeriques.columns:
    # Vérifier le nombre de valeurs uniques
    nombre_valeurs_uniques = marketing[colonne].nunique()
    if nombre_valeurs_uniques == 2:
        bin_cols.append(colonne)
        print(f"L'attribut '{colonne}' est binaire.")

# binary classes to categorical
marketing[bin_cols] = marketing[bin_cols].replace(0,'0').replace(1,'1')
marketing.drop('ID', axis=1, inplace=True)
print(marketing.info())

equilibre_classes = marketing['Response'].value_counts(normalize=True)
print("Répartition des classes dans 'Response':\n", equilibre_classes)

# Vérifier s'il y a des valeurs manquantes dans les données
valeurs_manquantes_par_colonne = marketing.isnull().sum()
print("Valeurs manquantes par colonne :\n", valeurs_manquantes_par_colonne)

# Récupérer seulement les valeurs numériques
data_num = marketing.select_dtypes(include='number')
print(data_num.info())

# Remplacer les valeurs manquantes par la moyenne pour les attributs numériques
imputer = SimpleImputer(strategy='mean')
data_num = imputer.fit_transform(data_num)
y = marketing['Response']
accuracy_score(lst_classif,lst_classif_names,data_num,y)

# Ddiscretize categorical values
data_cat = marketing.select_dtypes(exclude='number')

# Discretization with OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore")

encoder.fit(data_cat)
X_cat = encoder.transform(data_cat).toarray()
print("Score OneHotEncoder: ")
accuracy_score(lst_classif, lst_classif_names, X_cat, y)

# Concaténer les données numériques et catégorielles
X = np.concatenate((data_num, X_cat), axis=1)
print("Score Ensemble des données: ")
accuracy_score(lst_classif, lst_classif_names, X_cat,y)

# Appliquer le suréchantillonnage pour équilibrer les classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("score données équilibré: ")
accuracy_score(lst_classif, lst_classif_names, X_resampled, y_resampled)
