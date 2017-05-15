#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

# the training data (features_train, labels_train) have both "fast" and "slow"
# points mixed together--separate them so we can give them different colors
# in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

# initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()

reducer = 1
if reducer > 1:
    features_train = features_train[:len(features_train) / reducer]
    labels_train = labels_train[:len(labels_train) / reducer]

classifier_adaboost = AdaBoostClassifier(n_estimators=100)
classifier_random_forest = RandomForestClassifier(n_estimators=10,
                                                  max_depth=None,
                                                  min_samples_split=40,
                                                  random_state=0)

classifier_adaboost.fit(features_train, labels_train)
classifier_random_forest.fit(features_train, labels_train)

labels_predicted = classifier_adaboost.predict(features_test)
labels_predicted_random_forest = classifier_random_forest.predict(features_test)

accuracy_adaboost = accuracy_score(labels_test, labels_predicted)
accuracy_random_forest = accuracy_score(labels_test, labels_predicted_random_forest)

fast_count_adaboost = 0
slow_count_adaboost = 0
fast_count_random_forest = 0
slow_count_random_forest = 0

for v in labels_predicted:
    if v == 0:
        fast_count_adaboost += 1
    elif v == 1:
        slow_count_adaboost += 1

for v in labels_predicted_random_forest:
    if v == 0:
        fast_count_random_forest += 1
    elif v == 1:
        slow_count_random_forest += 1

print "adaboost accuracy:", accuracy_adaboost * 100, "%"
print "random forest accuracy:", accuracy_random_forest * 100, "%"
print "adaboost      --> fast:", fast_count_adaboost, "slow:", slow_count_adaboost
print "random forest --> fast:", fast_count_random_forest, "slow:", slow_count_random_forest

try:
    prettyPicture(classifier_adaboost, features_test, labels_test)
    # result saved as test.png
except NameError:
    pass
