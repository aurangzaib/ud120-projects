#!/usr/bin/python

import pickle
import numpy
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
numpy.random.seed(42)

# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
words_file = "text_learning/your_word_data.pkl"
authors_file = "text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))


# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(
    word_data,
    authors,
    test_size=0.1,
    random_state=42
    )

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()
# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

dt_classifier = DecisionTreeClassifier(min_samples_split=2)
dt_classifier.fit(features_train, labels_train)
labels_predicted = dt_classifier.predict(features_test)

_accuracy_ = accuracy_score(labels_test, labels_predicted)
_imp_ = dt_classifier.feature_importances_
_signature_index_ = (_imp_ >= max(_imp_)).nonzero()[0][0] # its numpy array and .index() doesn't work
_signature_word_ = vectorizer.get_feature_names()[_signature_index_]

print ("accuracy:", _accuracy_)
print ("training points:", len(features_train))
# remove from _imp_ --> elements which are > 0.2
print ("outliers:", filter(lambda condition: condition > 0.2, _imp_))
print ("signature number:", _signature_index_, "is:", max(_imp_))
print ("signature word:", _signature_word_)