"""
Things to do:
1. Read/Split the data from the file
2. Get 1 feature and 1 label
3. Split the data into train and test
4. Create a list of the ML models to be test
5. Train and Test each model
6. Get the best model
"""
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    # Read/Split the data from the file
    data = np.loadtxt('spamTrain1.csv', delimiter=',')

    # Get 1 feature and 1 label
    # TODO: Fix for a specific feature and label
    # TODO: Update with a loop to test all features and labels -> Maybe if not too big
    # TODO: Change 1st value in here to test a particular feature (0-29)
    features = data[:, 0]
    labels = data[:, -1]

    # Replace all -1 with the mean of the column
    features = np.where(features == -1, np.nan, features)
    mean = np.nanmean(np.where(features == -1, np.nan, features), axis=0)

    print(features, 'n', labels)

    # Split the data into train and test
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.5, random_state=1)

    # Create a list of the ML models to be test
    models = [Perceptron()]
    model_labels = ['Perceptron']
    best_accuracy = 0
    best_model = 0

    # Train and Test each model
    for i in range(len(models)):
        # Train the model
        model = models[i]
        model.fit(train_features, train_labels)

        # Test the model
        testOutputs = model.predict_proba(test_features)[:, 1]
        aucTestRun = roc_auc_score(test_labels, testOutputs)
        # print(f'Test set AUC: {aucTestRun}')

        # Update the best model
        if aucTestRun > best_accuracy:
            best_accuracy = aucTestRun
            best_model = i

    # Get the best model
    # print(f'Best model: {model_labels[best_model]}')
    # print(f'Best accuracy: {best_accuracy}')
