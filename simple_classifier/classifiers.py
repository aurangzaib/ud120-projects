"""
|__ Classification        --> Discrete Decision Boundaries, Predicts the class of features
                            |__ Naive Bayes
                            |__ Support Vector Machine
                            |__ Decision Tree
                            |__ Adaptive Boost --> ensemble Decision Trees
                            |__ Random Forest  --> ensember Decision Trees
|__ Regression            --> Continuous Output, Slope, Intercept, Predicts value of the features
|__ Cluster               --> Unsupervised Learning
                            |__ KMeans
                            |__ Input --> List of Features
                            |__ Output--> Predicted Clusters and Cluster Centers (Centroids)
|__ Feature Scale         --> MinMaxScaler --> 0...1
|__ Vectorizer            --> Vector of words' frequency from Text --> Bag Of Features --> Feature = Word
                            |__ List of Features
                            |__ Dictionary of Features
                            |__ List of Frequency of Features --> Frequency Vector
                            |__ CountVectorizer
                            |__ TfidfVectorizer --> term frequency inverse document frequency
|__ Text Classification   --> Stopwords, Stemmer, Vectorizer
|__ Feature Reduction     --> Reduce number of features and dimensions using PCA
                            |__ Principal Component Analysis
                            |__ Finding Principal Components
                            |__ Finding Variances 
                            |__ Using Principal Components in place of Features
                            |__ Compare Features & Principal Components usage in Image Recognition
|__ K Fold                  |__ Train/Test Features Split
"""
from __future__ import print_function
# import sys
import pickle
from sklearn.model_selection import train_test_split
from tools.feature_format import feature_format, target_feature_split

# sys.path.append("tools/")

dictionary = pickle.load(open("final_project/final_project_dataset_modified.pkl", mode="rb"))

# list the features you want to look at
# first item in the list will be the target feature
features_list = [
    # target
    "bonus",
    # features
    "long_term_incentive",
    "salary",
    "expenses"
]

raw_data = feature_format(dictionary, features_list, remove_any_zeroes=True)
target, features = target_feature_split(raw_data)
feature_train, feature_test, target_train, target_test = train_test_split(features,
                                                                          target,
                                                                          test_size=0.3,
                                                                          random_state=42)


class MachineLearningAlgorithms(object):
    @staticmethod
    def classify_nb(features_train, labels_train, features_test, labels_test):
        from sklearn.naive_bayes import GaussianNB  # --> gaussian naive bayes
        from sklearn.metrics import accuracy_score
        classifier = GaussianNB()
        classifier.fit(features_train, labels_train)
        labels_predicted = classifier.predict(features_test)
        # accuracy_score --> compare test and predicted labels
        accuracy = accuracy_score(labels_test, labels_predicted)
        print("accuracy:", accuracy * 100, "%", "--> naive bayes")
        return classifier

    @staticmethod
    def classify_svm(features_train, labels_train, features_test, labels_test):
        from sklearn.svm import SVC  # --> support vector classifier
        from sklearn.metrics import accuracy_score
        features_train_less = features_train[:int(len(features_train) / 10)]
        labels_train_less = labels_train[:int(len(labels_train) / 10)]
        classifier = SVC(kernel='rbf',
                         gamma=10.,  # sensitive to margins of near features
                         C=1000)  # more accurate less smooth
        # train on big data set --> high accuracy
        classifier.fit(features_train, labels_train)
        labels_predicted = classifier.predict(features_test)
        # train on small data set --> less accuracy
        classifier.fit(features_train_less, labels_train_less)
        labels_predicted_less = classifier.predict(features_test)
        # find accuracy by comparing test and predicted features
        accuracy = accuracy_score(labels_test, labels_predicted)
        accuracy_less = accuracy_score(labels_test, labels_predicted_less)
        print("accuracy:", accuracy * 100, "%", "--> support vector machine")
        print("accuracy:", accuracy_less * 100, "%", "--> support vector machine with 10% data")
        # print("F1 score:", f1_score(labels_test, labels_predicted))
        return classifier

    @staticmethod
    def classify_dt(features_train, labels_train, features_test, labels_test):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        # stop further split when node has only 40 samples remaining
        classifier = DecisionTreeClassifier(min_samples_split=40)
        classifier.fit(features_train, labels_train)
        labels_predicted = classifier.predict(features_test)
        accuracy = accuracy_score(labels_test, labels_predicted)
        print("accuracy:", accuracy * 100, "%", "--> decision tree")
        return classifier

    @staticmethod
    def classify_adaboost(features_train, labels_train, features_test, labels_test):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.metrics import accuracy_score
        # using 100 weak estimators --> ensemble
        classifier = AdaBoostClassifier(n_estimators=100)
        classifier.fit(features_train, labels_train)
        labels_predicted = classifier.predict(features_test)
        accuracy = accuracy_score(labels_test, labels_predicted)
        print("accuracy:", accuracy * 100, "%", "--> adaboost")
        return classifier

    @staticmethod
    def classify_random_forest(features_train, labels_train, features_test, labels_test):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        classifier = RandomForestClassifier(
            n_estimators=10, min_samples_split=40)
        classifier.fit(features_train, labels_train)
        labels_predicted = classifier.predict(features_test)
        accuracy = accuracy_score(labels_test, labels_predicted)
        print("accuracy:", accuracy * 100, "%", "--> random forest")
        return classifier

    @staticmethod
    def perform_linear_regression():

        """
        classifiers --> predict the class OF the features.
        regressions --> predict the target values FROM the features.

        classifiers --> output is discrete
        regressions --> output is continuous
                        --> also gives slope and intercept i.e. relation b/w feature and target

        classifiers --> evaluation using accuracy score
        regressions --> evaluation using sse, r-square, gradient-descent
        """
        from sklearn.linear_model import LinearRegression, Lasso

        import matplotlib.pyplot as plt

        import numpy as np
        import random
        """
        age --> feature 
        networth --> target
        """
        slope = 6.5
        age_train = []
        age_test = []
        networth_train = []
        networth_test = []
        for i in range(0, 100, 3):
            age_train.append([i])
        age_test.append([i + random.uniform(1, 2)])
        for i in range(len(age_train)):
            # y = mx
            networth_train.append([float(slope * age_train[i][0])])
        networth_test.append(
            [float(random.uniform(5, 8) * age_train[i][0])])

        regression_model = LinearRegression()
        lasso_model = Lasso(alpha=0.1)  # alpha=0 --> use linear regression
        regression_model.fit(age_train, networth_train)
        lasso_model.fit(age_train, networth_train)
        networth_predicted = regression_model.predict(age_test)
        networth_predicted_lasso = lasso_model.predict(age_test)
        print("predicted net worth for age 34:", regression_model.predict([[34]]))
        print("predicted net worth for age 34:", lasso_model.predict([[34]]))
        print("slope:", regression_model.coef_)
        print("intercept:", regression_model.intercept_)
        print("lasso slope:", lasso_model.coef_)
        print("lasso intercept:", lasso_model.intercept_)
        # low score --> over fitting
        # input, output
        print("r square error   : ", regression_model.score(age_test, networth_test))
        print("lasso r square error   : ", lasso_model.score(age_test, networth_test))
        # mean(square(prediction - actual))
        print("mean square error: ", np.mean(
            (networth_predicted - networth_test) ** 2))
        """
            SSE --> sum of squared error --> sum(actual - prediction) method to reduce regression errors
            SSE --> ordinary least square (OLS) & Gradient Descent
            Gradient Descent:
                yi = yi  + data_weight * ( xi - yi )
                yi = yi + smooth_weight * ( y(i-1) + y(i+1) - 2yi )
            Problem with SSE --> SSE increases by increasing the data points even if the fit is not getting worse.
            
            R Square Error:
            0(Worst Fit) < R < 1(Best Fit) 
        """
        plt.clf()
        plt.scatter(age_train, networth_train,
                    color="orange", label="training")
        plt.scatter(age_test, networth_test, color="red", label="test")
        plt.plot(age_test, networth_predicted, 'green', label="predictions")
        plt.legend(loc=2)
        plt.xlabel("age")
        plt.ylabel("net worth")
        plt.show()

    @staticmethod
    def kmeans_cluster(features):
        from sklearn.cluster import KMeans
        kmeans_cluster = KMeans(
            n_clusters=2,  # how many clusters
            n_init=10,  # how many times initialize with random centroids
            max_iter=400,  # max iterations if tolerance not reached
            tol=1e-4
        )
        kmeans_cluster.fit(features)  # features --> list of features
        predicted_target = kmeans_cluster.predict(
            features)  # no labels --> unsupervised
        centroids = kmeans_cluster.cluster_centers_  # predicted centroids
        return predicted_target, centroids

    @staticmethod
    def feature_rescale(arr):
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        rescaled = []
        list_max = max(arr)
        list_min = min(arr)
        if list_max == list_min:
            rescaled = [0.5 for row in range(len(arr))]
        else:
            for a in arr:
                rescaled.append(float(a - list_min) /
                                float(list_max - list_min))

        # matrix notation
        arr_1 = np.matrix('115.; 140.; 175.')
        # array notation
        arr_2 = np.array([[115.], [140.], [175.]])
        # list notation
        arr_3 = [115., 140., 175.]

        scalar = MinMaxScaler()
        # doesn't accept list, requires np array or matrix
        print(scalar.fit_transform(arr_2))
        return rescaled

    @staticmethod
    def count_vectorizer():
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        train_corpus = ["hello earth, from mars",
                        "she likes to eat", "lets get some food"]
        test_corpus = ["holly shit, look at the mars, there is some food food"]
        """
        1-gram --> each word is considered in isolation and they have no new meanings when used together
        """
        vectorizer = CountVectorizer(min_df=1, ngram_range=(0, 1))
        vectorizer.fit(train_corpus)
        # --> list of features
        vector_features = vectorizer.get_feature_names()
        vector_count = vectorizer.transform(test_corpus).toarray()
        # vector from the fit data
        print("features:", vector_features)
        # --> dictionary of features
        print("vocabulary:", vectorizer.vocabulary_)
        # count in the test text
        print("feature count:", vector_count)
        # vocabulary of particular word in train data
        print("vocabulary of mars:", vectorizer.vocabulary_.get("mars"))
        # assert if particular word is present in test data
        feature_index = vectorizer.vocabulary_.get("food")
        print("existence check", vector_count[:, feature_index])
        """
        n-gram
            with 1-gram we lose info of multi words like Chicago Bull
            even though it can have different meanings when used together.
            to resolve this, we use 2-grams.
            solves problem of Bag of Words problem (1-gram)
        """
        vectorizer = CountVectorizer(
            min_df=1, ngram_range=(1, 2), token_pattern=r'\b\w+\b')
        analyzer = vectorizer.build_analyzer()
        print("2 grams:", analyzer("Bull Chicago"))
        """
        tf-idf term weighting
            tf --> term frequency, idf --> inverse document frequency
            a, um, uh, is --> not of interest <-- they shadow the frequency of rare but interesting words.
            tf-idf --> re-weight the features in float suitable for classifier.
            it uses euclidean norms
            V-norm = V / ||V||2 = V / root(V1^2 + v2^2 + ...)
        """
        vectorizer = TfidfVectorizer()
        # compare with vector_count
        print("tf-idf count:", vectorizer.fit_transform(train_corpus).toarray())

    @staticmethod
    def stop_words():
        """
            stopwords --> words which have high frequency and low importance
            nltk --> natural language toolkit
            download nltk modules --> nltk.download('all', halt_on_error=False)
        """
        from nltk.corpus import stopwords
        from sklearn.feature_extraction.text import CountVectorizer
        corpus = [
            'This is a text which contains several trivial '
            'and some very important words. '
            'Lets remove the trivial words from it.']
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()
        sw = stopwords.words("english")
        print("total stopwords in german:", len(stopwords.words("german")))
        print("total stopwords in english:", len(sw))
        print("stopwords:", sw)

        print("before:", len(features))
        for s in sw:
            features = filter(lambda condition: condition != s, features)
        print("after:", len(features))

    @staticmethod
    def stemmering():
        """
        stemmer --> get the unique word from collection of words
        """
        # from nltk.stem.snowball import SnowballStemmer --> can also be used
        from nltk.stem.porter import PorterStemmer
        plurals = ['die', 'died', 'dying', 'dies', 'died',
                   'responsive', 'responsivity', 'unresponsive']
        stemmer = PorterStemmer()
        singles = [stemmer.stem(plural) for plural in plurals]
        print("plural:", plurals)
        print("singles:", singles)

    @staticmethod
    def text_classification():
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        sw = stopwords.words("english")
        corpus = 'This is a normal looking text. it contains a lot of insignificant texts. ' \
                 'Ok remove that text. also, ' \
                 'apply the stemming so that we can remove repeat repetition ' \
                 'repetitive health healthy healthiness sick sickness '' \
                ''die dying dies died ' \
                 'done do does go went gone going see sees seeing'
        # string --> list
        corpus_list = corpus.split()
        # removing stopwords
        for s in sw:
            corpus_list = filter(lambda condition: condition != s, corpus_list)
        # stemmering
        stemmer = PorterStemmer()
        corpus_list = [stemmer.stem(_corpus_) for _corpus_ in corpus_list]
        corpus = [' '.join(corpus_list)]
        # vector counts
        vectorizer = CountVectorizer(min_df=1, ngram_range=(0, 1))
        vectorizer.fit(corpus)
        vector_count = vectorizer.transform(corpus).toarray()
        feature_list = vectorizer.get_feature_names()
        for _index_ in range(len(feature_list)):
            print(feature_list[_index_], ":", vector_count[0][_index_])

    @staticmethod
    def principal_component_analysis(train=feature_train, test=feature_test):
        from sklearn.decomposition import PCA
        # how many components
        pca = PCA(n_components=2)
        # fit the pca
        pca.fit(train)
        # get principle components and variances
        pc_variance = pca.explained_variance_ratio_
        pc_values = pca.components_
        print("Variance:", pc_variance)
        print("Principal Components: ", pc_values)
        return pca.transform(train), pca.transform(test)

    @staticmethod
    def perform_k_fold_and_grid_search(data):
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.decomposition import PCA
        from sklearn.model_selection import GridSearchCV
        # split feature and target data
        labels_data, features_data = target_feature_split(data)
        features_train, labels_train, features_test, labels_test = [], [], [], []

        # split features and labels into train and test
        skf = StratifiedKFold(n_splits=3)
        for train_index, test_index in skf.split(features_data, labels_data):
            features_train = [features_data[index] for index in train_index]
            labels_train = [labels_data[index] for index in train_index]
            features_test = [features_data[index] for index in test_index]
            labels_test = [labels_data[index] for index in test_index]

        # perform principal components analysis and transform features into components
        pca = PCA(n_components=2)
        pca.fit(features_train)
        pca_train, pca_test = pca.transform(features_train), pca.transform(features_test)

        # dictionary of params for svm
        parameters = {
            'kernel': ('linear', 'rbf'),
            'C': [1, 10, 1000],
            'gamma': [10, 1000]
        }
        _svm_ = SVC()
        # grid search will find the best params
        svm_classifier = GridSearchCV(_svm_, parameters)

        # svm classifier for classification
        # principal components are used in place of features
        svm_classifier.fit(features_train, labels_train)
        print("best params:", svm_classifier.best_params_)
        labels_prediction = svm_classifier.predict(features_test)
        print("accuracy score: ", accuracy_score(labels_test, labels_prediction) * 100, "%")


ml = MachineLearningAlgorithms()
# SVM with features
ml.classify_svm(feature_train, target_train, feature_test, target_test)
pca_train_, pca_test_ = ml.principal_component_analysis(feature_train, feature_test)
# SVM with principle components
ml.classify_svm(pca_train_, target_train, pca_test_, target_test)

# k fold train/test splitting
ml.perform_k_fold_and_grid_search(raw_data)
