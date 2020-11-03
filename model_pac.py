import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def mainScript(news):
    filename = 'model.pkl'
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)

    with open(filename, 'rb') as ml_model:
        model = pickle.load(ml_model)


    tfidf_news =tfidf_vectorizer.transform(news)
    prediction = model.predict(news)
    predictedOutcome = prediction[0]

    return predictedOutcome
