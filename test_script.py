from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from zakar import *

def test_pred(text):
    stemmer = StemmerFactory().create_stemmer()
    # stem = stemmer.stem(text)

    vectorizer = TfidfVectorizer(max_features=1000, decode_error='ignore')
    # vectorizer.fit(text['text_stemming'])
    vectorizer.fit(text, stemmer)


    # class_pred = pickle_model.predict(vectorizer.transform([stem]))
    if text == 0:
        pred = 'Normal'
        status = 'text-success'
    elif text == 1:
        pred = 'Penipuan'
        status = 'text-danger'
    else:
        pred = 'Promo'
        status = 'text-warning'
    assert pred, status
