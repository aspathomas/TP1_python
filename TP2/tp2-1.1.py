#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Turn interactive plotting off
plt.ioff()

# # read input text and put data inside a data frame
worldDev = pd.read_csv('Data_World_Development_Indicators2.csv')

num_attributes = worldDev.shape[1]
print(f"Il y a {num_attributes} attributs.")
types_attributes = worldDev.dtypes
print("Types des attributs :\n", types_attributes)

missing_values = worldDev.isnull().sum()
missing_values_non_zero = missing_values[missing_values != 0]

country_codes = worldDev['Country Code']
country_names = worldDev['Country Name']

numeric_columns = worldDev.select_dtypes(include=['float64', 'int64']).columns
# Afficher les attributs et le nombre de valeurs manquantes seulement s'il y en a
if not missing_values_non_zero.empty:
    print("Nombre de valeurs manquantes par attribut:\n", missing_values_non_zero)

    # Remplacer les valeurs manquantes par la médiane
    imputer = SimpleImputer(strategy='median')
    worldDev = pd.DataFrame(imputer.fit_transform(worldDev[numeric_columns]), columns=numeric_columns)

# Appliquer StandardScaler
scaler = StandardScaler()
worldDev = pd.DataFrame(scaler.fit_transform(worldDev[numeric_columns]), columns=numeric_columns)

# plot instances on the first plan (first 2 factors) or 2nd plan
def plot_instances_acp(coord,df_labels,x_axis,y_axis, name):
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_xlim(-7,9) # limits must be manually adjusted to the data
    axes.set_ylim(-7,8)
    for i in range(len(df_labels.index)):
        plt.annotate(df_labels.values[i],(coord.iloc[i,x_axis],coord.iloc[i,y_axis]))
    plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
    plt.savefig(f"fig/acp_instances_{name}_axes_{str(x_axis)}_{str(y_axis)}")
    plt.close(fig)

# coord: results of the PCA 
# Appliquer l'ACP
pca = PCA()
acp = pca.fit_transform(worldDev)

# Créer un DataFrame avec les deux premiers composants principaux
pca_df_2d = pd.DataFrame(data=acp[:, :2], columns=['PC1', 'PC2'])

# Créer un DataFrame avec les troisième et quatrième composants principaux
pca_df_3d = pd.DataFrame(data=acp[:, 2:4], columns=['PC3', 'PC4'])

plot_instances_acp(pca_df_2d, country_codes, 0, 1, "1_2")
plot_instances_acp(pca_df_3d, country_codes, 0, 1, "3_4")




# compute correlations between factors and original variables
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# plot correlation_circles
def correlation_circle(components,var_names,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    minx = -1
    maxx = 1
    miny = -1
    maxy = 1
    axes.set_xlim(minx,maxx)
    axes.set_ylim(miny,maxy)
    # label with variable names
    # ignore first variable (instance name)
    for i in range(0, components.shape[1]):
        axes.arrow(0,
                   0,  # Start the arrow at the origin
                   components[i, x_axis],  #0 for PC1
                   components[i, y_axis],  #1 for PC2
             head_width=0.01,
             head_length=0.02)

        plt.text(components[i, x_axis] + 0.05,
                 components[i, y_axis] + 0.05,
                 var_names[i])
    # axes
    plt.plot([minx,maxx],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[miny,maxy],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

# ignore 1st 2 columns: country and country_code
correlation_circle(loadings, worldDev.columns, 0, 1)




# print centroids associated with several countries
lst_countries=[]
# centroid of the entire dataset
# est: KMeans model fit to the dataset
print(est.cluster_centers_)
for name in lst_countries:
    num_cluster = est.labels_[y.loc[y==name].index][0]
    print('Num cluster for '+name+': '+str(num_cluster))
    print('\tlist of countries: '+', '.join(y.iloc[np.where(est.labels_==num_cluster)].values))
    print('\tcentroid: '+str(est.cluster_centers_[num_cluster]))

