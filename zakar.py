from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pytest

@pytest.fixture(scope='module')
def text():
    data = ['hallo selamat pagi', 'silahkan transfer ke nomor rekening', 'selamat anda berhasil memnangkan']
    pred = [0, 1, 2]

    return {
        'data': data,
        'pred': pred,
    }

def test_pred(text):
    stemmer = StemmerFactory().create_stemmer()
    vectorizer = TfidfVectorizer(max_features=1000, decode_error='ignore')
    vectorizer.fit(text, stemmer)

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
