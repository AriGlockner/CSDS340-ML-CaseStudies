# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.
@author: Kevin S. Xu
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    if trainFeatures is None or trainLabels is None:
        km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
        return km.fit_predict(testFeatures)

    # Otherwise use the labels to train a random forest classifier
    # and predict the labels of the test data
    rf = RandomForestClassifier(n_estimators=100, random_state=100)
    rf.fit(trainFeatures, trainLabels)
    return rf.predict(testFeatures)


def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures=testFeatures, numVessels=20, trainFeatures=trainFeatures, trainLabels=trainLabels)


def convertToLabels(features):
    sog_index = 3
    cog_index = 4

    if features is not None:
        for i in range(len(features)):
            sog = features[i, sog_index]
            cog = (features[i, cog_index] / 10.0) * math.pi / 180.0

            vx = sog * math.cos(cog)
            vy = sog * math.sin(cog)

            features[i, sog_index] = vx
            features[i, cog_index] = vy

    return features


def testTrained(features, labels):
    print('With Train Features/Labels:')
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    '''
    # Convert features to labels
    X_train = convertToLabels(X_train)
    X_test = convertToLabels(X_test)
    '''

    # Run prediction algorithms and check accuracy
    # Prediction with specified number of vessels
    numVessels = np.unique(y_train).size

    # Supervised prediction --> training data is used
    predVesselsWithK = predictWithK(X_test, numVessels, X_train, y_train)

    ariWithK = adjusted_rand_score(y_test, predVesselsWithK)

    # Prediction without specified number of vessels
    # Unsupervised prediction --> training data is unused
    predVesselsWithoutK = predictWithoutK(X_test, X_train, y_train)

    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(y_test, predVesselsWithoutK)

    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: ' + f'{ariWithoutK}')

    return predVesselsWithK, predVesselsWithoutK


def testUntrained(features, labels):
    print('No trainFeatures or trainLabels given:')
    # Run prediction algorithms and check accuracy
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size

    # Unsupervised prediction --> training data is unused
    predVesselsWithK = predictWithK(features, numVessels)

    ariWithK = adjusted_rand_score(labels, predVesselsWithK)

    # Prediction without specified number of vessels
    # Unsupervised prediction --> training data is unused
    predVesselsWithoutK = predictWithoutK(features)

    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)

    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: ' + f'{ariWithoutK}')

    return predVesselsWithK, predVesselsWithoutK


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks

    data = loadData('set1.csv')
    features = data[:, 2:]
    labels = data[:, 1]

    # %% Plot all vessel tracks with no coloring
    plotVesselTracks(features[:, [2, 1]])
    plt.title('All vessel tracks')

    # %% Run prediction algorithms and check accuracy

    predVesselsWithK, predVesselsWithoutK = testUntrained(features, labels)
    predVesselsWithK, predVesselsWithoutK = testTrained(features, labels)

    # %% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:, [2, 1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:, [2, 1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:, [2, 1]], labels)
    plt.title('Vessel tracks by label')
