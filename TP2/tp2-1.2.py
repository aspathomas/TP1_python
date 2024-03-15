#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv


# read input data
dataset = []
with open('data/grocery.csv', 'r') as read_obj: 
    csv_reader = csv.reader(read_obj) 
    list_of_csv = list(csv_reader) 
    # remove header
    list_of_csv.pop(0)
    # remove trailing empty fields
    dataset = list(map(lambda l: [e for e in l if e != ''], list_of_csv))
    
from mlxtend.preprocessing import TransactionEncoder

# convert lists of string values into one-hot vectors
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
grocery = pd.DataFrame(te_ary, columns=te.columns_)
print(grocery.head())


# option to show all itemsets
pd.set_option('display.max_colwidth',None)

frequent_itemsets = apriori(grocery, min_support=0.002, use_colnames=True)
print(len(frequent_itemsets))

# Regroupement des itemsets fréquents par leur taille
grouped_itemsets = frequent_itemsets.groupby(by=frequent_itemsets['itemsets'].apply(len))

# Calcul de la fréquence relative en fonction du nombre d'items
frequency_by_size = grouped_itemsets.size() / len(frequent_itemsets)

# Affichage de la fréquence relative en fonction du nombre d'items
print("Nombre d'Items\tFréquence")
for size, frequency in frequency_by_size.items():
    print(f"{size}\t\t{frequency:.4f}")

# Construction des règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Affichage du nombre de règles obtenues
print("Nombre de règles d'association obtenues:", len(rules))
# Première règle
print("Première règle: ", rules.iloc[0])

best_rule_confidence = rules.sort_values(by='confidence', ascending=False).iloc[0]
print("Meilleure règle pour la métrique confidence:")
print(best_rule_confidence)

best_rule_lift = rules.sort_values(by='lift', ascending=False).iloc[0]
print("Meilleure règle pour la métrique lift:")
print(best_rule_lift)

# Filtrer les règles pour ne conserver que celles avec du yaourt et du café comme antécédents
filtered_rules = rules[rules['antecedents'].apply(lambda x: {'yogurt', 'coffee'}.issubset(x))]

# Afficher les conséquents des règles filtrées
for idx, row in filtered_rules.iterrows():
    print("Produits associés avec yaourt et café:", row['consequents'])
