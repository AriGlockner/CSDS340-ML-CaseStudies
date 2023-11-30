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

    # testFeatures = transformFeatures(testFeatures)
    # foo

    # If training data is not given, use k-means clustering to predict the labels
    if trainFeatures is None or trainLabels is None:
        km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100, tol=2e-4)
        return km.fit_predict(testFeatures)

    # Otherwise use the labels to train a random forest classifier
    # and predict the labels of the test data
    trainFeatures = scaler.fit_transform(trainFeatures)
    #trainFeatures = transformFeatures(trainFeatures)

    # Train random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=100, min_samples_split=50, criterion='log_loss')
    rf.fit(trainFeatures, trainLabels)
    return rf.predict(testFeatures)


def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures=testFeatures, numVessels=20, trainFeatures=trainFeatures, trainLabels=trainLabels)


def transformFeatures(features):
    """
    Transform features to improve clustering performance. The initial features are:
    Timestamp - hh:mm:ss
    Latitude - degrees
    Longitude - degrees
    SOG - speed over ground
    COG - course over ground
    :param features: the features to transform
    :return: the transformed features
    """
    # TODO: Implement feature transformation
    new_labels = ['latitude', 'longitude', 'x_velo', 'y_velo', 'hours', 'minutes', 'seconds', 'bearing', 'SOW',
                  'lee_way']
    new_features = [[], [], [], [], [], [], []]

    for i in range(features.shape[0]):
        # Add latitude and longitude
        new_features[0].append(features[i, 0])
        new_features[1].append(features[i, 1])

        # Convert SOG and COG to x and y velocities
        sog = features[i, 0]
        cog = features[i, 1]
        new_features[2].append(sog * math.cos(cog))
        new_features[3].append(sog * math.sin(cog))

        # Convert timestamp to hours, minutes, and seconds
        timestamp = features[i, 2]
        hours = timestamp // 3600
        minutes = (timestamp % 3600) // 60
        seconds = timestamp % 60
        new_features[4].append(hours)
        new_features[5].append(minutes)
        new_features[6].append(seconds)

    return np.array(new_features).T


def testTrained(features, labels):
    print('With Train Features/Labels:')
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
