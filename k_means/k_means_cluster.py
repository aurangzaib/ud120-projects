#!/usr/bin/python
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("tools/")
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from feature_format import feature_format, target_feature_split


def Draw(prediction, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """
    # plot each cluster with a different color
    # add more colors for drawing more than five clusters
    # colour is picked based on the cluster number
    # clusters are predicted by k-means
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(prediction):
        plt.scatter(features[ii][0], features[ii][1],
                    color=colors[prediction[ii]])
    # if you like, place red stars over points that are POI
    if mark_poi:
        for ii, pp in enumerate(prediction):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii]
                            [1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# load in the dict of dicts containing all the data on each person in the
# dataset
data_dict = pickle.load(open("final_project/final_project_dataset.pkl", "r"))
data_dict.pop("TOTAL", 0)  # outlier removed
# the input features we want to use
# can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"

# add 3rd feature in features_list and compare the results
# after adding 3rd feature, total 4 data points exchanged their positions
# in the plot
features_list = [poi,
                 feature_1,
                 feature_2]
# splitting dictionary to list
# list containing poi(target), feature1, feature2
data = feature_format(data_dict, features_list)
# splitting list to further lists
# separate poi(target) from feature1, feature2
poi, finance_features = target_feature_split(data)
# feature scaling
# after scaling --> 0...1
# run k-means with and without scaling and comparing the results
# some of the data points will be clustered in different cluster after re-scaling.
# in this case, we may not need scaling
# but when we are using salary and from_messages as features then scaling
# is critical

# finance_features = MinMaxScaler().fit_transform(finance_features)

exercised_stock_options_values = []
salary_values = []
# change to f1, f2, f3 --> for 3 features
for f1, f2 in finance_features:
    if f1 != 0:
        salary_values.append(f1)
    if f2 != 0:
        exercised_stock_options_values.append(f2)
    plt.scatter(f1, f2)
plt.show()

print "exercise stock - max:", max(exercised_stock_options_values)
print "exercise stock - min:", min(exercised_stock_options_values)
print "salary - max:", max(salary_values)
print "salary - min:", min(salary_values)
"""
I don't know should I divide data into train and test or not.
they used the same data for train and test
"""
# feature_train, feature_test = train_test_split(finance_features, test_size=0.5, random_state=42)

kmeans_cluster = KMeans(n_clusters=2,  # how many clusters
                        n_init=10,  # try 10 initial random centroids
                        max_iter=300,  # max iterations if tolerance not reached
                        tol=0.0001)  # tolerance
# note that train labels are not given. as it's unsupervised learning.
kmeans_cluster.fit(finance_features)
# prediction on the same data as training
# predicts --> which data point belongs to which cluster
prediction = kmeans_cluster.predict(finance_features)
# cluster centers
# gives feature values of the centroids
print "total clusters:", len(kmeans_cluster.cluster_centers_)
print "cluster centers:\n", kmeans_cluster.cluster_centers_
"""
compare plot 1 with plot 2
plot 1 --> data without clustering
plot 2 --> clustered data, notice the colour of the points
"""
try:
    Draw(prediction,
         finance_features,
         poi, mark_poi=False,
         # rename the "name" parameter when you change the number of features
         # so that the figure gets saved to a different file
         name="clusters-2-features.pdf",
         f1_name=feature_1,
         f2_name=feature_2)
except NameError:
    print "no predictions object named prediction found, no clusters to plot"