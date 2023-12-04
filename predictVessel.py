# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, OPTICS, cluster_optics_dbscan

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Preprocess data
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # Fit a HDBSCAN model
    clusterer = hdbscan.HDBSCAN(min_cluster_size=numVessels, min_samples=20, gen_min_span_tree=True)
    clusterer.fit(testFeatures)

    # Ensure that no more than numVessels clusters are found
    unique_labels, counts = np.unique(clusterer.labels_, return_counts=True)
    if len(unique_labels) > numVessels:
        # Identify the largest numVessels clusters
        largest_clusters = np.argsort(counts[unique_labels])[::-1][:numVessels]

        # Assign points outside the largest clusters to a new cluster label (-1)
        mask = np.isin(clusterer.labels_, largest_clusters, invert=True)
        clusterer.labels_[mask] = -1

    print(f'Number of Vessels: {numVessels} \n Number of Clusters: {len(np.unique(clusterer.labels_))}')
    return clusterer.labels_



def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)
    # TODO: Adjust arbitrary values
    # Arbitrary values for eps and min_samples
    dbscan = DBSCAN(eps=0.5, min_samples=20)
    return dbscan.fit_predict(testFeatures)


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]

    #%% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    