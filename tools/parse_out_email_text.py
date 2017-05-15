#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        """
    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()
    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        # apply stemming --> unify variations of a single word
        # string --> list
        text_string_list = text_string.split()
        # create stemmer
        stemmer = SnowballStemmer("english")
        # apply stemmer
        text_string_list = [stemmer.stem(_text_) for _text_ in text_string_list]
        # list --> string
        text_string = " ".join(text_string_list)

        words = text_string
    return words


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print (text)


if __name__ == '__main__':
    main()
