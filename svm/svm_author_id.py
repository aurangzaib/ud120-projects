#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.append("tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
# create gaussian naive bayes classifier
features_train, features_test, labels_train, labels_test = preprocess()
"""
    SVM are much slower to train compared to naive bayes classifier

    but one should always run different classifiers on each data set 
    to identify which classifier performs better in which data set

    normally classifiers are sensitive to the parameters
    so parameter tuning can achieve dramatically different results for
    same classifier on same data set.

    its same as image processing. opencv.

    compare time taking of naive bayes and support vector machine :

    svm training time: 5.947 s
    svm prediction time: 7.074 s
    support vector accuracy: 0.965301478953

    bayes training time: 0.184 s
    bayes prediction time: 0.253 s
    naive bayes accuracy: 0.961319681456

    setting C parameter (controls smoothness vs accuracy) to 1000 (more accuracy but less smooth decision surface) 
    --> svm and rbf kernel:
    svm training time: 0.137 s
    svm prediction time: 1.229 s
    support vector accuracy: 0.892491467577
"""
reduce_size = 10
# reducing number of data points -- SVM is very slow for large data set
# providing just 1% of the features
# accuracy is 88.8% which is very good for 1% features (data set)
features_train = features_train[:int(len(features_train) / reduce_size)]
labels_train = labels_train[:int(len(labels_train) / reduce_size)]
print ("number of features in data set:", len(features_train[0]))
# changing percentile --> SelectPercentile --> email_preprocess -->
# controls # of features

# create a support vector classifier
classifier_svm = SVC(kernel="rbf",  # try linear, rbf
                     C=10000)  # try 1, 10, 1000, 10000

time_svm = time()
classifier_svm.fit(features_train, labels_train)
print ("svm training time:", round(time() - time_svm, 3), "s")

time_predict_svm = time()
labels_predicted = classifier_svm.predict(features_test)
print ("svm prediction time:", round(time() - time_predict_svm, 3), "s")

# find the accuracy of the predicted labels against the test labels
# never check the accuracy against labels_train
accuracy_svm = accuracy_score(labels_test, labels_predicted)

"""
currently, our labels only contain information about the class
seeing prediction for the class for a different elements (data points)
"""
chris_found = 0
sara_found = 0
for v in labels_predicted:
    # class 1 --> chris
    if v == 1:
        chris_found += 1
    # class 2 --> sara
    elif v == 0:
        sara_found += 1

print ("number of emails from chris:", chris_found)
print ("number of emails from sara:", sara_found)
print ("support vector accuracy:", accuracy_svm)
