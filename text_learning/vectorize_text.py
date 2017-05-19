#!/usr/bin/python
import os
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("../tools/")
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""
from_sara = open("../text_learning/from_sara.txt", "rb")  # rb --> read as bytes
from_chris = open("../text_learning/from_chris.txt", "rb")

from_data = []
word_data = []

# preprocess all the emails
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = os.path.join('..', path[:-1])
        # paths of the emails   
        email = open(path, "rb")
        # texts of the email
        # remove punctuation and apply stemming
        email_text = parseOutText(email)
        from_data.append(0 if name is "sara" else 1)
        # remove these words from the email text
        # **** these words affect the feature importance and overfit in find_signature.py ****
        for _word_ in ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf", "houectect", "houston"]:
            email_text = email_text.replace(_word_, "")
        # save email_text in word_data
        word_data.append(email_text)
        # close the stream of the email
        email.close()

# create text-frequency inverse document frequency vectorizer
vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
# fit data and get the usage count of the features
vector_count = vectorizer.fit_transform(word_data).toarray()
# get the list of features names
features_list = vectorizer.get_feature_names()
# close the files
from_sara.close()
from_chris.close()
# write the results in directory --> text_learning
pickle.dump(word_data, open("../text_learning/your_word_data.pkl", "wb"))  # w --> write
pickle.dump(from_data, open("../text_learning/your_email_authors.pkl", "wb"))
