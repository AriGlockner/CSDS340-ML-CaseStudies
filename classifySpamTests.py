# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import base, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
from evaluateClassifier import tprAtFPR

random_state = 1


def aucCV(features, labels, classifier=RandomForestClassifier(random_state=random_state)):
    # model = RandomForestClassifier()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'),
                          classifier)
    return cross_val_score(model, features, labels, cv=10, scoring='roc_auc')


def predictTest(trainFeatures, trainLabels, testFeatures, classifier=RandomForestClassifier()):
    # model = RandomForestClassifier()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'), classifier)
    model.fit(trainFeatures, trainLabels)

    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    return model.predict_proba(testFeatures)[:, 1]


def fill_empty_values(data):
    """
    Fill empty values with the mean of the column
    :param data:
    :return:
    """
    # Fill missing values with the mean of the column
    imp = SimpleImputer(missing_values=-1, strategy='mean')
    imp.fit(data)
    return imp.transform(data)


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt('spamTrain1.csv', delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    features = fill_empty_values(data[:, :-1])
    labels = data[:, -1]

    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features, labels)))

    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]

    # The classifiers we are going to test
    classifiers = [RandomForestClassifier(random_state=random_state, criterion='gini'),
                   RandomForestClassifier(random_state=random_state, criterion='entropy'),
                   RandomForestClassifier(random_state=random_state, criterion='log_loss'),
                   BaggingClassifier(random_state=random_state),
                   ExtraTreesClassifier(random_state=random_state),
                   AdaBoostClassifier(random_state=random_state, learning_rate=0.1),
                   AdaBoostClassifier(random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state, loss='log_loss'),
                   GradientBoostingClassifier(random_state=random_state, loss='exponential')]

    # The labels for the classifiers
    classifier_labels = ['Random Forest - gini', 'Random Forest - entropy', 'Random Forest - log loss',
                         'Bagging/Bootstrap', 'Extra Trees', 'Adaboost - 0.1 learning rate', 'AdaBoost',
                         'Gradient Boosting - log loss', 'Gradient Boosting - exponential']

    # Best accuracy and TPR
    best_auc = 0
    best_tpr = 0
    best_auc_model = None
    best_tpr_model = None

    # Train and Test each model
    for i in range(len(classifiers)):
        print(f'\nClassifier: {classifier_labels[i]}')

        # Data for storing the average AUC and TPR at FPR = 0.01
        n_iter = 10
        auc_avg = 0
        tpr_avg = 0

        # Calculate the sums of the AUC and TPR at FPR = 0.01
        for j in range(n_iter):
            classifier = copy.deepcopy(classifiers[i])

            # Train the model
            testOutputs = predictTest(trainFeatures, trainLabels, testFeatures, classifier)

            # Calculate the AUC
            auc = np.mean(aucCV(testFeatures, testLabels, classifier))
            auc_avg += auc

            # Calculate the TPR at FPR = 0.01
            tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, 0.01)
            tpr_avg += tprAtDesiredFPR

        # Calculate the average AUC and TPR at FPR = 0.01
        auc_avg /= n_iter
        tpr_avg /= n_iter

        # Update the best auc model
        if auc_avg > best_auc:
            best_auc = auc_avg
            best_auc_model = classifier_labels[i]

        # Update the best tpr model
        if tpr_avg > best_tpr:
            best_tpr = tpr_avg
            best_tpr_model = classifier_labels[i]

        # Print the average AUC and TPR at FPR = 0.01 for the current model
        print(f'Average test set AUC: {auc_avg}')
        print(f'Average TPR at FPR = 0.01: {tpr_avg}')

    # Summary
    print(f'\nBest AUC: {best_auc} for {best_auc_model}')
    print(f'Best TPR: {best_tpr} for {best_tpr_model}')
