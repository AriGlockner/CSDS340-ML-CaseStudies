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
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # Transform features to improve clustering performance
    # testFeatures = transformFeatures(testFeatures)
    # foo
    # If training data is not given, use k-means clustering to predict the labels
    if trainFeatures is None or trainLabels is None:
        testFeatures = transformFeatures(testFeatures)

        # Extract the timestamp from the features
        timestamp = testFeatures[:, 0]
        other_features = testFeatures[:, 1:]

        # DBSCAN clustering on the timestamp
        clustering = DBSCAN(eps=0.5, min_samples=10, metric='euclidean', algorithm='auto', leaf_size=30, p=None,
                            n_jobs=None)
        # OPTICS(min_samples=10, xi=.05, min_cluster_size=.05) # Not bad, not best

        clustering.fit(timestamp.reshape(-1, 1))
        agg_labels = clustering.labels_

        # Convert the labels to one-hot encoding
        agg_labels = np.eye(numVessels)[agg_labels]

        # Concatenate the one-hot encoding with the other features
        testFeatures = np.concatenate((agg_labels, other_features), axis=1)

        km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
        return km.fit_predict(testFeatures)

    # Otherwise use the labels to train a random forest classifier
    # and predict the labels of the test data
    trainFeatures = scaler.fit_transform(trainFeatures)
    # trainFeatures = transformFeatures(trainFeatures)

    # Train random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=100, min_samples_split=50, criterion='log_loss')
    rf.fit(trainFeatures, trainLabels)
    return rf.predict(testFeatures)


def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    # Arbitrarily assume 20 vessels
    return predictWithK(testFeatures=testFeatures, numVessels=20, trainFeatures=trainFeatures, trainLabels=trainLabels)


def transformFeatures(old_features, numVessels=20):
    """
    Transform features to improve clustering performance. The initial features are:
    Timestamp - hh:mm:ss
    Latitude - degrees
    Longitude - degrees
    SOG - speed over ground
    COG - course over ground
    :param old_features: the features to transform
    :param numVessels: the number of vessels
    :return: the transformed features
    """
    '''
    for i in range(old_features.shape[0]):
        # Convert the SOG and COG to x and y components
        sog = old_features[i, 3]
        cog = old_features[i, 4]
        old_features[i, 3] = sog * np.cos(cog)
        old_features[i, 4] = sog * np.sin(cog)

    return old_features
    '''
    newFeatures = np.zeros((old_features.shape[0], 5))
    for i in range(old_features.shape[0]):
        # Move the timestamp, latitude, and longitude to the front
        newFeatures[i, 0] = old_features[i, 0]
        newFeatures[i, 1] = old_features[i, 1]
        newFeatures[i, 2] = old_features[i, 2]

        # Convert the SOG and COG to x and y components
        sog = old_features[i, 3]
        cog = old_features[i, 4]
        newFeatures[i, 3] = sog * np.cos(cog)
        newFeatures[i, 4] = sog * np.sin(cog)

    return newFeatures


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
    from utils import loadData, plotVesselTracks, convertTimeToSec

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
