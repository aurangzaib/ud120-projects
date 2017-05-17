#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot

sys.path.append("tools/")
from feature_format import feature_format, target_feature_split

# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "rb"))
features = ["salary", "bonus"]

# remove the outlier
# uncomment it and see the effect on graph
data_dict.pop("TOTAL", 0)
data = feature_format(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
