from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

class Classifier:
    def __init__(self,train_tweets,train_labels,test_tweets,test_labels,mode):
        self.mode = mode
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_tweets = train_tweets
        self.test_tweets = test_tweets
        features_train, features_test = Classifier.extract_features(train_tweets, test_tweets)
        Classifier.Tuning(features_train,train_labels)
        if self.mode == 1:
            SVMpredictLabels = Classifier.SVMClassifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(SVMpredictLabels, test_labels))
        elif self.mode == 2:
            RFpredictLabels = Classifier.RandomForest_Classifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(RFpredictLabels, test_labels))
        elif self.mode == 3:
            LRpredictLabels = Classifier.LogisticalRegression_Classifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(LRpredictLabels, test_labels))
        elif self.mode == 4:
            NBpredictLabels = Classifier.NaiveBayesClassifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(NBpredictLabels, test_labels))
        elif self.mode == 5:
            KNNpredictLabels = Classifier.KNNClassifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(KNNpredictLabels, test_labels))
        elif self.mode == 6:
            TreepredictLabels = Classifier.TreeClassifier(features_train, train_labels, features_test)
            print(Classifier.getAccuracy(TreepredictLabels, test_labels))
        else:
            print("Please select a valid classifier")


    def SVMClassifier(features_train, labels_train, features_test):
        # bn3ml training 3la train data (bn build our method 3aleha )
        # f b3ml object mn classifier bt3i w b3den fit de bt3ml train lal data bt3ty
        # tol---> nesbt el error el masbo7 beha el lw wsl 3ndha aw 2al yw2f w my7rksh el separator
        # random_state is the seed used by the random number generator
        # linear SVC da shbh precepton
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(features_train, labels_train)
        # b predict b2a 3la test data bt3ty 3shn agib accuracy bt3t classifier da
        X = clf.predict(features_test)
        return X


    def RandomForest_Classifier(features_train, labels_train, features_test):
        clf = RandomForestClassifier(n_estimators=1600,min_samples_split = 2, min_samples_leaf = 4,max_features = 'sqrt',         
        max_depth = 110)
        clf.fit(features_train, labels_train)
        return clf.predict(features_test)


    def LogisticalRegression_Classifier(features_train, labels_train, features_test):
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf.fit(features_train, labels_train)
        return clf.predict(features_test)


    def NaiveBayesClassifier(features_train, labels_train, features_test):
        clf = MultinomialNB()
        clf.fit(features_train, labels_train)
        return clf.predict(features_test)


    def KNNClassifier(features_train, labels_train, features_test):
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(features_train, labels_train)
        return clf.predict(features_test)


    def getAccuracy(output_labels, actual_labels):
        return accuracy_score(output_labels, actual_labels)


    def calculateConfusionMatrix(output_labels, actual_labels):
        cm = confusion_matrix(actual_labels, output_labels)
        print(cm)

    def extract_features(ourTweets,ourTestTweets):
        vectorizer = TfidfVectorizer()
        # return value ---> position of the word , index of tweet , tfidf value of the word.
        X = vectorizer.fit_transform(ourTweets)
        Y = vectorizer.transform(ourTestTweets)
        return X,Y;

    def TreeClassifier(trainData,train_labels,testData):
        tree_clf = DecisionTreeClassifier(max_depth=40)
        tree_clf.fit(trainData, train_labels)
        predict = tree_clf.predict(testData)
        return predict
    
    def Tuning(trainFeatures,trainLabel):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        #cv---> 3dad el folds , n_iter kam combination ygrbha , estimator eh hwa classifier el 7y3ml tuning , n_jobs 3dad el cores               el 7ysh3'lha
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 60, cv = 3, verbose=2,              
        random_state=42, n_jobs=-1)

        # Fit the random search model
        rf_random.fit(trainFeatures, trainLabel)
        print(rf_random.best_params_)