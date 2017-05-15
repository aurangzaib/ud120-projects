#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math
from tools.feature_format import feature_format

# read strings from file and returns the original object hierarchy
# we are reading enron + finance dataset
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# ../final_project/poi_names.txt --> contains poi of only enron folks
# ../final_project/final_project_dataset.pkl --> contains data of enron + finance

print "\n\nfeature: "
for feature in enron_data["SKILLING JEFFREY K"]:
    print feature
print "\n\n"

print "persons:", len(enron_data)
print "features/person:", len(enron_data["GRAMM WENDY L"])

# total number of persons of interest
number_of_poi = 0
for person in enron_data:
    if enron_data[person]["poi"] is True:
        number_of_poi += 1
print "POIs:", number_of_poi
print "stocks   - James Prentice:", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "email    - Wesley Colwell --> POI:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# print "\n\ntotal payments: "
# for person in enron_data: print person, ":", enron_data[person]["total_payments"]
# print "\n\n"

print "payments - skilling:", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "payments - kenneth:", enron_data["LAY KENNETH L"]["total_payments"]
print "payments - fastow:", enron_data["FASTOW ANDREW S"]["total_payments"]

# known salaries and email addresses
known_salaries = 0
known_emails = 0
payments_nan = 0
poi_payments_nan = 0
for person in enron_data:
    if not math.isnan(float(enron_data[person]["salary"])):
        known_salaries += 1
    if enron_data[person]["email_address"] != "NaN":
        known_emails += 1
    if math.isnan(float(enron_data[person]["total_payments"])):
        payments_nan += 1
        if enron_data[person]["poi"] is True:
            print enron_data[person]
            poi_payments_nan += 1

print "known salaries:", known_salaries
print "known emails:", known_emails
print "unknown payments:", float(payments_nan) / len(enron_data) * 100, "%"
poi_payments_nan = float(poi_payments_nan) / len(number_of_poi) * 100 if poi_payments_nan > 0 else poi_payments_nan
print "unknown payments poi:", poi_payments_nan, "%"

"""
if machine learning algorithm were to use with total payments as a feature,
it will associate NaN with non POIs as all POIs have the total_payments.
this classification is not true
"""
aa = feature_format(enron_data, ["salary", "total_payments"])

"""
most of non-POIs come from financial statements
most POIs were added by hand from enron data --> we have no data for them.
algorithm can be fooled into thinking that POIs are those which have no data.

so algorithm is learning false definition of POIs --> person with NaN features is POI.
when we run this learner on new data which has known data for POIs then it will fail.

we should be very careful when we are using data from several sources.
1 solution could be use features only which are COMMON IN ALL DATA. like email addresses.

we should be careful about introducing new features when we have data from different sources.
this is one of examples of introducing bias and mistakes.
"""
