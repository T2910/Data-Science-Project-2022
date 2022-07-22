import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



def split_train_test(X, Y, test_size):
    N_TOTAL_ROWS = X.shape[0]
    N_TEST = int(test_size * N_TOTAL_ROWS)

    test_indexes = sorted(random.sample(range(0, N_TOTAL_ROWS), N_TEST))

    X_test = X.iloc[test_indexes, :]
    Y_test = Y.iloc[test_indexes]
    X_train = X.drop(test_indexes, axis=0)
    Y_train = Y.drop(test_indexes, axis=0)

    return X_train, Y_train, X_test, Y_test


def train_classifier(classifier, X_train, Y_train, X_test, Y_test):
    # Fit classifier to train set
    classifier.fit(X_train, Y_train)

    # Make predictions for test set
    predictions = classifier.predict(X_test)

    # Check accuracy
    n_correct_classifications = 0
    expected = Y_test.to_numpy()
    for i in range(len(predictions)):
        if (predictions[i]==expected[i]):
            n_correct_classifications += 1

    acc = n_correct_classifications/len(predictions)
    print(f"Accuracy on test set: {acc}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, predictions))

    print("Classification Report")
    print(classification_report(Y_test, predictions))

    print("\n----------------------------------------\n\n")

    return classifier


def store_classifier(classifier, filename):
    file = open(filename, 'wb')
    pickle.dump(classifier, file)
    file.close()


# ------------------------------------------------------------------------ #
# Train classifiers using training and test set
# ------------------------------------------------------------------------ #
def train():
    # Load data
    DATA_ALL = pd.read_csv("clean_important.csv")

    # Divide data into Train/Test set
    Y_ALL = DATA_ALL["category"]
    X_ALL = DATA_ALL.drop(["category"], axis=1)
    X_train, Y_train, X_test, Y_test = split_train_test(X_ALL, Y_ALL, 0.2)


    # Gradient Boost
    print("Gradient Boosting Classifier\n")
    classifier = GradientBoostingClassifier(n_estimators=25, learning_rate=0.1, max_features=10, max_depth=10)
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_gbc.sav")

    # Decision Tree (not used)
    # print("Decision Tree Classifier\n")
    # classifier = DecisionTreeClassifier(max_depth=2)
    # classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    # store_classifier(classifier, "models/model_dtc.sav")

    # Bagging Ensemble
    print("Bagging Ensemble\n")
    classifier = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, max_samples=100, bootstrap=True, n_jobs=-1)
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_bag.sav")

    # Random Forests
    print("Random Forests\n")
    classifier = RandomForestClassifier(n_estimators=50, max_leaf_nodes=16, n_jobs=-1)
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_rfc.sav")

    # AdaBoost
    print("AdaBoost\n")
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, algorithm="SAMME.R", learning_rate=0.5)
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_ada.sav")

    # KMeans (not used because very poor accuracy: 14%)
    # print("KMeans\n")
    # classifier = KMeans(n_clusters=8)
    # classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    # store_classifier(classifier, "models/model_kmeans.sav")

    # KNeighborsClassifier
    print("K Neighbors\n")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_knn.sav")

    # MLPClassifier (not used because moderate accuracy: 55%)
    # print("MLP\n")
    # classifier = MLPClassifier(solver='lbfgs', hidden_layer_sizes=100, max_iter = 500)
    # classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    # store_classifier(classifier, "models/model_mlp.sav")

    # Logistic Regression
    print("Logistic Regression\n")
    classifier = LogisticRegression()
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_lrc.sav")

    # Gaussian Naive Bayes
    print("Gaussian Naive Bayes\n")
    classifier = GaussianNB()
    classifier = train_classifier(classifier, X_train, Y_train, X_test, Y_test)
    store_classifier(classifier, "models/model_gnb.sav")


# ------------------------------------------------------------------------ #
# Predict depression for given data
# ------------------------------------------------------------------------ #
def predict(data):
    filenames = ['models/model_gbc.sav', "models/model_dtc.sav", "models/model_bag.sav", "models/model_rfc.sav", "models/model_ada.sav", "models/model_knn.sav", "models/model_lrc.sav"]
    classifiers = [pickle.load(open(file, 'rb')) for file in filenames]

    predictions = pd.DataFrame(classifiers[0].predict(data))

    print(classifiers)
    for clf in classifiers[1:]:
        predictions = pd.concat([predictions, pd.DataFrame(clf.predict(data))], axis=1)

    majority_vote = predictions.mode(axis=1)[0].to_numpy()
    print('MAJORITY VOTE')
    print(majority_vote)

    return majority_vote


def main(args):
    # Set seed
    random.seed(43)

    # Training
    if (args.train):
        train()

    # Prediction
    if (args.predict):
        if not args.data:
            print("No data provided.\n")
            return

        data = pd.read_csv(args.data)
        predictions = pd.DataFrame(data={'prediction': predict(data)})
        predictions = predictions.replace(0.0, 'No Depression')
        predictions = predictions.replace(1.0, 'Moderate Depression')
        predictions = predictions.replace(2.0, 'Severe Depression')

        data = pd.concat([data, predictions], axis=1)
        data.to_csv(args.out)


    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for detecting possible depression for given data.')
    parser.add_argument('--train', default=False, action='store_true', help='Whether to train the models using the existing data.')
    parser.add_argument('--predict', default=False, action='store_true', help='Whether to predict the depression for new data. New data must be provided with the "--data" flag. For this, the trained models must exist, so training has to be done before.')
    parser.add_argument('--data', default=None, help='The path of the .csv file to use for predicting the depression. It is assumed that this already has the same columns as the existing data that was used to train the models.')
    parser.add_argument('--out', default='out.csv', help='The path of the output file.')

    args = parser.parse_args()

    main(args)
