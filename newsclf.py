import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle


def get_data():
    dataFrame = pd.read_csv('news.csv')
    dataFrame.describe()
    dataFrame.info()
    dataFrame.shape

    X = dataFrame.text
    y = dataFrame.label

    return X, y


def training():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=7)

    tfidf_Xtrain, tfidf_Xtest = Vectorize(X_train, X_test)

    Pac = PassiveAggressiveClassifier(C = 0.5, random_state = 5)

    Pac.fit(tfidf_Xtrain, y_train)

    Pac_acc = Pac.score(tfidf_Xtest, y_test)

    print(Pac_acc)

    y_pred = Pac.predict(tfidf_Xtest)

    Pac_accuracy = accuracy_score(y_test, y_pred)

    print(Pac_accuracy)

    conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])

    print(conf_matrix)

    clf_report = classification_report(y_test, y_pred)

    print(clf_report)

    makePickleFile(Pac)



def makePickleFile(classifier):
    filename = "model.pkl"
    with open(filename, 'wb') as modelfile:
        pickle.dump(classifier, file)


def Vectorize(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    return tfidf_train, tfidf_test
