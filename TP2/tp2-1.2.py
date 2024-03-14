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
with open('grocery.csv', 'r') as read_obj: 
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

# frequent_itemsets = apriori(...)

# rules=association_rules(frequent_itemsets, ...

# select rules with more than 2 antecedents
# rules.loc[map(lambda x: len(x)>2,rules['antecedents'])]


