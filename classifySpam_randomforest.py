# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from evaluateClassifier import tprAtFPR


def aucCV(features,labels,classifier=RandomForestClassifier()):
    # model = RandomForestClassifier()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'),
                          classifier)
    return cross_val_score(model,features,labels,cv=10,scoring='roc_auc')


def predictTest(trainFeatures,trainLabels,testFeatures,classifier=RandomForestClassifier()):
    # model = RandomForestClassifier()
    model = make_pipeline(SimpleImputer(missing_values=-1, strategy='mean'), RandomForestClassifier())
    model.fit(trainFeatures, trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    return model.predict_proba(testFeatures)[:,1]


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
    data = np.loadtxt('spamTrain1.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = fill_empty_values(data[:,:-1])
    labels = data[:,-1]
    
    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features,labels)))
    
    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2,:]
    trainLabels = labels[0::2]
    testFeatures = features[1::2,:]
    testLabels = labels[1::2]

    classifiers = [RandomForestClassifier(), BaggingClassifier(), ExtraTreesClassifier(), AdaBoostClassifier()]
    classifier_labels = ['Random Forest', 'Bagging/Bootstrap', 'Extra Trees', 'AdaBoost']

    for i in range(len(classifiers)):
        print(f'\nClassifier: {classifier_labels[i]}')

        # Train the model
        testOutputs = predictTest(trainFeatures, trainLabels, testFeatures, classifiers[i])

        # Calculate the AUC
        print("Test set AUC: ", np.mean(aucCV(testFeatures, testLabels, classifiers[i])))

        # Calculate the TPR at FPR = 0.01
        tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels,testOutputs,0.01)
        print("TPR at FPR = 0.01: ", tprAtDesiredFPR)

    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    