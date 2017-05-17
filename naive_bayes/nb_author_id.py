#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

sys.path.append("tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
"""
    computing the accuracy of the gaussian naive bayesian classifier

    we should train the classifier on different data samples to generalize it
    otherwise the algorithm will be 'over fit' for a given data
    so, we try to save 10% of train data and use it as part of test data without first training on it.

    train data --> data on which classifier is trained using fit (features, labels)
    test data  --> data on which the prediction of classifier is tested

    so after training (fitting), we provide classifier features and predict the labels.
    
    to report results of the classifier, run it on the test data and report the predictions. 
    
    compare time taking of naive bayes and support vector machine
    
    --- accuracy ---
    
    on 100% data set, nb has 97.3% accuracy
    on 100% data set, nb has 99% accuracy
    
    on 1% data set, nb has 90% accuracy 
    on 1% data set, svm has 88% accuracy
    
    on 10% data set, with rbf ans C=1000, svm has 89% accuracy
"""
reduce_size = 100
# reducing number of data points -- SVM is very slow for large data set
# providing just 1% of the features
# accuracy is 88.8% which is very good for 1% features (data set)
features_train = features_train[:int(len(features_train) / reduce_size)]
labels_train = labels_train[:int(len(labels_train) / reduce_size)]
print ("number of features in data set:", len(features_train[0]))
# changing percentile --> SelectPercentile --> email_preprocess -->
# controls # of features

# create gaussian naive bayes classifier
classifier_nb = GaussianNB()

time_bayes = time()
# train classifier by fitting training data
classifier_nb.fit(features_train, labels_train)
print ("bayes training time:", round(time() - time_bayes, 3), "s")
time_predict_bayes = time()
# predict the labels by providing test features
labels_predicted = classifier_nb.predict(features_test)
print ("bayes prediction time:", round(time() - time_predict_bayes, 3), "s")

# find the accuracy of the predicted labels against the test labels
# never check the accuracy against labels_train
accuracy = accuracy_score(labels_test, labels_predicted)
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
print ("naive bayes accuracy:", accuracy)
