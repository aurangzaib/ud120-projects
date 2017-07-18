#!/usr/bin/python
""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary
"""
from simple_classifier.prep_terrain_data import makeTerrainData
from simple_classifier.class_vis import prettyPicture, output_image
from simple_classifier.classifiers import MachineLearningAlgorithms

features_train, labels_train, features_test, labels_test = makeTerrainData()

# the training data (features_train, labels_train) have both "fast" and "slow" points mixed
# in together--separate them so we can give them different colors in the scatterplot,
# and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.

_nb = MachineLearningAlgorithms.classify_nb(features_train, labels_train, features_test, labels_test)
_dt = MachineLearningAlgorithms.classify_dt(features_train, labels_train, features_test, labels_test)
_svm = MachineLearningAlgorithms.classify_svm(features_train, labels_train, features_test, labels_test)
_adaboost = MachineLearningAlgorithms.classify_adaboost(features_train, labels_train, features_test, labels_test)
_random_forest = MachineLearningAlgorithms.classify_random_forest(features_train, labels_train, features_test, labels_test)

prettyPicture(_nb, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
