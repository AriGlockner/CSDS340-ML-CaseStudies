"""
Things to do:
1. Read/Split the data from the file
2. Get 1 feature and 1 label
3. Split the data into train and test
4. Create a list of the ML models to be test
5. Train and Test each model
6. Get the best model
"""
import math

import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

import evaluateClassifier


if __name__ == "__main__":
    # Read/Split the data from the file
    data = np.loadtxt('spamTrain1.csv', delimiter=',')

    # Get 1 feature and 1 label
    # TODO: Fix for a specific feature and label
    # TODO: Update with a loop to test all features and labels -> Maybe if not too big
    # TODO: Change 1st value in here to test a particular feature (0-29)
    # Initially feature1 = 0 and feature2 = 1
    feature1 = 0
    feature2 = 1
    features = data[:, feature1:feature2]
    labels = data[:, -1]

    # Replace all -1 with the mean of the column
    mean = 0.0
    number_of_values = 0
    for i in range(len(features)):
        if features[i] == -1:
            features[i] = np.nan
        else:
            mean += features[i]
            number_of_values += 1

    mean /= number_of_values

    for i in range(len(features)):
        if np.isnan(features[i]):
            features[i] = mean

    '''features = np.where(features == -1, np.nan, features)
    mean = np.nanmean(np.where(features == -1, np.nan, features), axis=0)'''

    # Split the data into train and test
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.5, random_state=1)

    # Create a list of the ML models to be test
    C = 2.0
    models = [Perceptron(), GaussianNB(), SVC(kernel='linear', C=C), SVC(kernel='poly', C=C), SVC(kernel='rbf', C=C),
              SVC(kernel='sigmoid', C=C), DecisionTreeClassifier(), KNeighborsClassifier()]
    model_labels = ['Perceptron', 'Naive Bayes', 'Adaline', 'Logistic Regression', 'SVM - Linear', 'SVM - poly',
                    'SVM - rbf', 'SVM - sigmoid', 'Decision Tree', 'K-Nearest Neighbors']
    best_accuracy = 0
    best_tpr = 0
    best_model = 0

    # Train and Test each model
    for i in range(len(models)):
        # Train the model
        model = models[i]
        model.fit(train_features.copy(), train_labels.copy())

        # TODO: DO NOT USE PREDICT FUNCTION FOR SCIKIT-LEARN
        # Test the model
        print(f'\nModel: {model_labels[i]}')
        # testOutputs = evaluateClassifier.predictTest(train_features.copy(), train_labels.copy(), test_features.copy())

        # TODO: Use Random Forest or Boosting here to generate multiple models
        testOutputs = 0 # fill this in properly

        # Calculate the AUC
        aucTestRun = roc_auc_score(test_labels, testOutputs)
        aucTestRun = max(aucTestRun, 1 - aucTestRun)
        print(f'Test set AUC: {aucTestRun}')

        # Calculate the TPR
        tprAtDesiredFPR, fpr, tpr = evaluateClassifier.tprAtFPR(test_labels, testOutputs, 0.01)
        print(f'TPR at FPR = 0.01: {tprAtDesiredFPR}')

        # Update the best model
        if aucTestRun > best_accuracy:
            best_accuracy = aucTestRun
            best_model = i
            best_tpr = tprAtDesiredFPR
        if aucTestRun == best_accuracy:
            if tprAtDesiredFPR > best_tpr:
                best_accuracy = aucTestRun
                best_model = i
            elif tprAtDesiredFPR == best_tpr:
                print('Same accuracy and TPR')

    # TODO: Figure out why all the models have the same output
    # Get the best model
    print('\nFeature 1: ', feature1, '\tFeature 2: ', feature2)
    print(f'Best model: {model_labels[best_model]}')
    print(f'Best accuracy: {best_accuracy}')
