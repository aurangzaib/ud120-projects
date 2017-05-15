#!/usr/bin/python
"""
    Clean away the 10% of points that have the largest
    residual errors (difference between the prediction
    and the actual net worth).

    Return a list of tuples named cleaned_data where 
    each tuple is of the form (age, net_worth, error).
"""


def outlierCleaner(predictions,  # predictions of net worth
                   ages,  # feature
                   net_worths  # target
                   ):
    error_list = []
    for i in range(len(predictions)):
        # create list of tuple(index, error)
        error_list.append(tuple((i, (predictions[i][0] - net_worths[i][0]) ** 2)))

    cleaned_data = []
    # sort by error where error --> tup[1]
    # remove 10% max error element
    # create list of (ages, net_worth, error)
    for a_tuple in sorted(error_list, key=lambda tup: tup[1])[:int(len(ages) * 0.9)]:
        a_index = a_tuple[0]
        cleaned_data.append(tuple((ages[a_index][0], net_worths[a_index][0], a_tuple[1])))
    return cleaned_data
