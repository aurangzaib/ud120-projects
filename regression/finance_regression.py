#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

import sys
import pickle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

sys.path.append("tools/")
from feature_format import feature_format, target_feature_split

dictionary = pickle.load(open("../final_project/final_project_dataset_modified.pkl", "r"))

# list the features you want to look at -- first item in the list will be the "target" feature
features_list = [
    "bonus",  # target
    "salary"  # feature -- use salary, long_term_incentive and other features to compare score
]

"""
long term incentives have better relation with bonus than the salaries
we find it by comparing r square scores while using both features as input and bonus as target
"""

data = feature_format(dictionary, features_list, remove_any_zeroes=True)
target, features = target_feature_split(data)
# training-testing split needed in regression, just like classification
feature_train, feature_test, target_train, target_test = train_test_split(features,
                                                                          target,
                                                                          test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

# we are trying to predict bonus using salary
# feature --> salary, long_term_incentive or any other feature --> input
# target --> bonus --> output

reg = LinearRegression()
reg.fit(feature_train, target_train)
target_prediction = reg.predict(feature_test)
intercept_prediction = reg.intercept_
slope_prediction = reg.coef_

print "intercept:", intercept_prediction
print "slope:", slope_prediction

r_square_result = reg.score(feature_test, target_test) # input, output comparison for test
r_square_train_result = reg.score(feature_train, target_train) # input, output comparison for train
# gives the score -1.48 which is a very bad result
# it shows linear regression is not able to find any relation
# between input (salary) and target (bonus)
print "r square result:", r_square_result
print "r square result on train data:", r_square_train_result

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

try:
    plt.plot(feature_test, target_prediction)
    # using test data -- without outliers for fitting
    # observe the change in slope
    reg.fit(feature_test, target_test)
    print "intercept without outliers:", reg.intercept_
    print "slope without outliers:", reg.coef_
    plt.plot(feature_train, reg.predict(feature_train), color="b")
except NameError:
    pass
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
