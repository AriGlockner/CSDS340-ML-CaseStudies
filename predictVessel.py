# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.
@author: Kevin S. Xu
"""
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

weight_distances = []


def compute_weight_distances(features):
    """
    Sets the weight distances for the features.
    :param features: the features to set the weight distances for
    :return: Nothing. Sets the weight distances for the features.
    """
    weight_distances.clear()

    # Calculate the distance between each pair of features
    for i in range(1, len(features[0])):
        weight_distances.append(max(features[:, i]) - min(features[:, i]))


def predictWithK(testFeatures, numVessels, trainFeatures=None, trainLabels=None):
    compute_weight_distances(testFeatures)
    # Unsupervised prediction, so training data is unused
    scaler = StandardScaler()
    testFeatures = scaler.fit_transform(testFeatures)

    # Transform features to improve clustering performance
    testFeatures = transformFeatures(testFeatures)

    # If training data is not given, use k-means clustering to predict the labels
    if trainFeatures is None or trainLabels is None:
        km = KMeans(n_clusters=numVessels, init='k-means++', n_init=10, random_state=100)
        return km.fit_predict(testFeatures)

    # Otherwise use the labels to train a random forest classifier
    # and predict the labels of the test data
    trainFeatures = scaler.fit_transform(trainFeatures)
    trainFeatures = transformFeatures(trainFeatures)

    # Train random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=100)
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
    # Extract the timestamp feature from the features and convert to seconds
    timestamp = np.array([(time % 60) + (((time // 60) % 60) * 60) + (time // 3600) * 3600
                          for time in old_features[:, 0:]])

    # Use Agglomerative Clustering to cluster the timestamps
    timestamp_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=60)
    timestamp_labels = timestamp_clustering.fit_predict(timestamp)

    # Create a dictionary to store the clusters
    timestamp_clusters = {}

    # Add each timestamp to its cluster
    for i in range(len(timestamp_labels)):
        if timestamp_labels[i] not in timestamp_clusters:
            timestamp_clusters[timestamp_labels[i]] = [timestamp[i]]
        else:
            timestamp_clusters[timestamp_labels[i]].append(timestamp[i])

    '''
    We are now going to use nested Agglomerative Clustering. The first level of clustering will be done on the
    timestamp clusters. The second level of clustering will be done on the remaining features. The number of clusters
    for the second level of clustering will be the number of vessels. The features will be clustered by the mean
    distance between the features and the cluster center. The features will be weighted by the weight distances.
    '''

    # Create an array to store the transformed features
    transformed_features = np.zeros_like(old_features)

    # Iterate through each timestamp cluster
    for cluster_label, timestamps_in_cluster in timestamp_clusters.items():
        # Filter features that belong to the current timestamp cluster
        cluster_mask = np.isin(timestamp, timestamps_in_cluster)
        cluster_features = []
        for i in range(len(cluster_mask)):
            if cluster_mask[i].any():
                cluster_features.append(old_features[i])

        # Extract the remaining features for the second level of clustering
        remaining_features = cluster_features

        # Use Agglomerative Clustering for the second level on the remaining features
        feature_clustering = AgglomerativeClustering(n_clusters=numVessels)
        feature_labels = feature_clustering.fit_predict(remaining_features)

        # Iterate through each feature cluster and assign the mean value to its members
        for i in range(numVessels):
            feature_cluster_mask = feature_labels == i
            feature_cluster_mean = np.mean(cluster_features, axis=0)
            transformed_features[feature_cluster_mask] = feature_cluster_mean

    return transformed_features


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
    # predVesselsWithK, predVesselsWithoutK = testTrained(features, labels)

    # %% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:, [2, 1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:, [2, 1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:, [2, 1]], labels)
    plt.title('Vessel tracks by label')
