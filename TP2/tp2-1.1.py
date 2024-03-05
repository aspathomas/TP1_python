#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Turn interactive plotting off
plt.ioff()

# # read input text and put data inside a data frame
# worldDev = 



# plot instances on the first plan (first 2 factors) or 2nd plan
def plot_instances_acp(coord,df_labels,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(20,20))
    axes.set_xlim(-7,9) # limits must be manually adjusted to the data
    axes.set_ylim(-7,8)
    for i in range(len(df_labels.index)):
        plt.annotate(df_labels.values[i],(coord[i,x_axis],coord[i,y_axis]))
    plt.plot([-7,9],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-7,8],color='silver',linestyle='-',linewidth=1)
    plt.savefig('fig/acp_instances_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)

# coord: results of the PCA 
# plot_instances_acp(coord,y,0,1)




# compute correlations between factors and original variables
# loadings = acp.components_.T * np.sqrt(acp.explained_variance_)

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
correlation_circle(loadings,covid.columns[2:],0,1)





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

