#!/usr/bin/python
"""
    Clean away the 10% of points that have the largest
    residual errors (difference between the prediction
    and the actual net worth).

    Return a list of tuples named cleaned_data where 
    each tuple is of the form (age, net_worth, error).
"""


def outlier_cleaner(predicted_labels,
                    features,
                    labels
                    ):
    error_list = []
    for i in range(len(predicted_labels)):
        # create list of tuple(index, error)
        error_list.append(tuple((i, (predicted_labels[i][0] - labels[i][0]) ** 2)))

    cleaned_data = []
    # sort by error where error --> tup[1]
    # remove 10% max error element
    # create list of (ages, net_worth, error)
    for _tuple_ in sorted(error_list, key=lambda tup: tup[1])[:int(len(features) * 0.9)]:
        _index_ = _tuple_[0]
        cleaned_data.append(tuple((features[_index_][0], labels[_index_][0], _tuple_[1])))
    return cleaned_data
