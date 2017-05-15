def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """
    computing the accuracy of the guassian bayesian classifer

    we should train the classifier on different data to generalize it
    otherwise the algorithm will overfit for a given data
    so, we try to save 10% of train data ...
    ...and use it as part of test data without first training on it.

    train data -- data on which classifier is trained using fit (features, labels)
    test data -- data on which the prediction of classifier is tested

    so after training (fit), we provide classifier fesatues and predict the labels.
    should not use same data for training and testing -- creates overfit problem.abs

    to report results of the classifier, run it on the test data and report the predictions. 
    """

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    # create a classifier
    clf = GaussianNB()

    # train classifier by fitting training data
    clf.fit(features_train, labels_train)

    # predict the labels by providing test features
    labels_predicted = clf.predict(features_test)

    # find the accuracy of the predicted labels
    accuracy = accuracy_score(labels_test, labels_predicted)

    return accuracy
