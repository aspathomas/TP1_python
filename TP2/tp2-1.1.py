#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# Turn interactive plotting off
plt.ioff()

# # read input text and put data inside a data frame
worldDev = pd.read_csv('data/Data_World_Development_Indicators2.csv')

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
    worldDev_numeric = pd.DataFrame(imputer.fit_transform(worldDev[numeric_columns]), columns=numeric_columns)

# Appliquer StandardScaler
scaler = StandardScaler()
worldDev_numeric = pd.DataFrame(scaler.fit_transform(worldDev_numeric[numeric_columns]), columns=numeric_columns)

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
acp = pca.fit_transform(worldDev_numeric)

# Créer un DataFrame avec les deux premiers composants principaux
pca_df_2d = pd.DataFrame(data=acp[:, :2], columns=['PC1', 'PC2'])
0
# Créer un DataFrame avec les troisième et quatrième composants principaux
pca_df_3d = pd.DataFrame(data=acp[:, 2:4], columns=['PC3', 'PC4'])

plot_instances_acp(pca_df_2d, country_codes, 0, 1, "1_2")
plot_instances_acp(pca_df_3d, country_codes, 0, 1, "3_4")

for i in range(len(country_codes.index)):
        print("pays: ", country_codes.values[i], " ACP_0: ",pca_df_2d.iloc[i,0]," ACP_1: ", pca_df_2d.iloc[i,1])

for i in range(len(country_codes.index)):
        print("pays: ", country_codes.values[i], " ACP_2: ",pca_df_3d.iloc[i,0]," ACP_3: ", pca_df_3d.iloc[i,1])



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
    plt.title(f"cercle de corrélation {x_axis} et {y_axis} facteurs acp")
    plt.plot([minx,maxx],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[miny,maxy],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

correlation_circle(loadings, worldDev.columns, 0, 1)
correlation_circle(loadings, worldDev.columns, 2, 3)


# Question 6
# Liste pour stocker les valeurs de coefficient R2
r2_scores = []

# Faire varier le nombre de clusters (k) de 2 à 20
for k in range(2, 21):
    # Effectuer le clustering avec KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(worldDev_numeric)
    
    # Calculer le coefficient R2 (score silhouette)
    silhouette_avg = silhouette_score(worldDev_numeric, kmeans.labels_)
    r2_scores.append(silhouette_avg)

print(r2_scores)
# Tracer la variation du coefficient R2 en fonction de k
plt.figure(figsize=(10, 6))
plt.plot(range(2, 21), r2_scores, marker='o', linestyle='-')
plt.title('Variation du coefficient R2 en fonction de k')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Coefficient R2')
plt.xticks(np.arange(2, 21, step=1))
plt.grid(True)
plt.savefig('fig/varition_R2')

#Question 7
# Fixer le nombre de clusters à 8
k = 8

# Effectuer le clustering avec KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(worldDev_numeric)

# Identifier les indices des pays dans le dataframe original
france_index = worldDev.index[worldDev['Country Name'] == 'France'][0]
mexico_index = worldDev.index[worldDev['Country Name'] == 'Mexico'][0]
bulgaria_index = worldDev.index[worldDev['Country Name'] == 'Bulgaria'][0]

# Identifier les clusters auxquels appartiennent la France, le Mexique et la Bulgarie
france_cluster = kmeans.labels_[france_index]
mexico_cluster = kmeans.labels_[mexico_index]
bulgaria_cluster = kmeans.labels_[bulgaria_index]

# Afficher les profils des groupes contenant la France, le Mexique et la Bulgarie
group_france = worldDev[kmeans.labels_ == france_cluster]
group_mexico = worldDev[kmeans.labels_ == mexico_cluster]
group_bulgaria = worldDev[kmeans.labels_ == bulgaria_cluster]


# Concatenate the DataFrames for France, Mexico, and Bulgaria
group_stats = pd.concat([group_france.describe(), group_mexico.describe(), group_bulgaria.describe()], axis=0)

# Save the concatenated DataFrame to a CSV file
group_stats.to_csv('fig/group_statistics.csv', sep=';', decimal=',')

# Calculer la matrice de liaison avec la méthode de liaison choisie (par exemple, 'ward')
Z = linkage(worldDev_numeric, method='ward')

# Afficher le dendrogramme
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Dendrogramme Hiérarchique Ascendant')
plt.xlabel('Indices des Échantillons')
plt.ylabel('Distance Euclidienne')
plt.savefig('fig/dendrogramme')