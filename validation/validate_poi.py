#!/usr/bin/python
from __future__ import print_function
import pickle
import sys

sys.path.append("../tools/")
from feature_format import feature_format, target_feature_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = feature_format(data_dict, features_list)
labels, features = target_feature_split(data)

features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=42
                                                                            )
classifier = DecisionTreeClassifier(min_samples_split=2)
classifier.fit(features_train, labels_train)
predicted_labels = classifier.predict(features_test)
accuracy = accuracy_score(labels_test, predicted_labels)
print("accuracy:", accuracy)
