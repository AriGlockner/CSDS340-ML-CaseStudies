# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    # Preprocess data
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # Initialize number of clusters to be greater than the number of vessels
    n_clusters = numVessels + 1
    pred_labels = None
    eps = 0.236

    # Iterate until the number of clusters is equal to the number of vessels
    while n_clusters > numVessels:
        # Fit a DBSCAN model
        model = DBSCAN(eps=eps, min_samples=20)
        pred_labels = model.fit_predict(testFeatures)
        n_clusters = np.unique(pred_labels).size

        # Decrease the epsilon value if the number of clusters is too large
        eps += 0.01

    # Return the predicted labels
    return pred_labels


def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Preprocess data
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # Values determined by testing and iteration
    model = DBSCAN(eps=0.236, min_samples=20)
    return model.fit_predict(testFeatures)


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
    