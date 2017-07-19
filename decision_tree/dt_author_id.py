#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.metrics import accuracy_score

sys.path.append("tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
reduce_size = 1
# reducing number of data points -- SVM is very slow for large data set
# providing just 1% of the features
# accuracy is 88.8% which is very good for 1% features (data set)
features_train = features_train[:int(len(features_train) / reduce_size)]
labels_train = labels_train[:int(len(labels_train) / reduce_size)]

"""
number of data points --> len(features_train) --> rows
number of features    --> len(features_train[0]) --> columns
"""
classifier = DecisionTreeClassifier(min_samples_split=40,
                                    presort=True
                                    # speed up the finding of best splits in fitting
                                    # on large data set, it may slow down the training process
                                    )
print("number of features in data set:", len(features_train[0]))
# changing percentile --> SelectPercentile --> email_preprocess --> controls # of features

# min_samples_split --> 2 has 90% accuracy
# min_samples_split --> 30,40,45 has 91% accuracy
train_time = time()
classifier.fit(features_train, labels_train)
print("dt training time:", round(time() - train_time, 3), "s")

predict_time = time()
labels_predicted = classifier.predict(features_test)
print("dt prediction time:", round(time() - predict_time, 3), "s")

accuracy_tree = accuracy_score(labels_test, labels_predicted)

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

print("number of emails from chris:", chris_found)
print("number of emails from sara:", sara_found)
print("decision tree accuracy:", accuracy_tree)
