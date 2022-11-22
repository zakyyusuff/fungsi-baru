from zakar import *
from sklearn.base import TransformerMixin
import spacy

class StopWordRemovalTransformer(TransformerMixin):
    """Removes stop words and punctuation from text (English only).
    """ 

    def tokenizeText(self, sample):
        tokens = self._nlp(sample)
        # tokens = [tok.text for tok in tokens if tok not in STOPLIST]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
        return ' '.join(tokens)

    def transform(self, X, **transform_params):
        X = [iText(text) for text in X] 
        return [self.tokenizeText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

class LemmatizeTransformer(TransformerMixin):  
    """Transforms words to lemmatized form (English only).
    """ 
    def __init__(self):
        super().__init__()
        self._nlp = spacy.load('en_core_web_sm')


    def lemmatizeText(self, sample):
        tokens = self._nlp(sample)
        tokens = [tok.lemma_.lower().strip() for tok in tokens]
        return ' '.join(tokens)

    def transform(self, X, **transform_params):
        X = [iText(text) for text in X] 
        return [self.lemmatizeText(text) for text in X]
 

    def fit(self, X, y=None, **fit_params):
        return self


class DocEmbeddingVectorizer(TransformerMixin):  
    """Convert a collection of text documents to a matrix containing a document embedding.
    """ 
    def __init__(self):
       super().__init__()
       self._nlp = spacy.load("en_vectors_web_lg")

    def transform(self, X, **transform_params):
        X = [iText(text) for text in X]
        return [self._nlp(text).vector for text in X]
 

    def fit(self, X, y=None, **fit_params):
        return self
