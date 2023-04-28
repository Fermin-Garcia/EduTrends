from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import prepare
def get_corr_heatmap(train):
    '''
    This function will display a heatmap of the potential correlations between variables in 
    our dataset
    '''
    # get the correlation values
    corr_matrix = train.corr()
    # create a plot
    plt.figure(figsize=(10,10))
    # plot a heatmap of the correlations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # add a title
    plt.title('Heat Map of Correlation')
    # display the plot
    plt.show()

def best_kmeans(data,k_max):
    '''
    EXAMPLE USEAGE:
    data = scaled_train[['alcohol', 'quality']]
    best_kmeans(data,k_max=10)

    This function will produce an elbow graph with clusters
    '''
    # create empty list variables to store results
    means = []
    inertia = []
    # cycle through our desired amount of k's
    for k in range(1, k_max):
        # create a KMeans object with current k
        kmeans = KMeans(n_clusters=k)
        # fit the kmeans object to our data
        kmeans.fit(data)
        # store the kmeans object in our means list
        means.append(k)
        # store the inertia for current k in the inertia list
        inertia.append(kmeans.inertia_)
        # create a figure
        fig =plt.subplots(figsize=(10,5))
        # plot the current k and inertia
        plt.plot(means,inertia, 'o-')
        # add axis labels
        plt.xlabel('means')
        plt.ylabel('inertia')
        # remove gridlines
        plt.grid(True)
        # display the plot
        plt.show()

def apply_kmeans(data,k):
    '''
    This function will create a clusters based on the given variables and data
    '''
    # create a kmeans object with k clusters
    kmeans = KMeans(n_clusters=k)
    # fit the kmeans object on our data
    kmeans.fit(data)
    # store the clustered data as a new column
    data[f'k_means_{k}'] = kmeans.labels_
    # return the modified dataset
    return data

