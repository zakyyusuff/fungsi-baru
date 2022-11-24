from functools import partial
from class_fungsi import StopWordRemovalTransformer, LemmatizeTransformer, DocEmbeddingVectorizer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import numpy as np
from urllib.request import urlopen, urlretrieve
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
import tarfile
import pandas as pd
import os.path
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import string

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

def iText(text):
    label = {'label': ['spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham']} 
    data = {'text': ['AngeBot ist, geheim!', 'Klicke geheim Link', 'Geheim, sport link', 'Spiel SPORT heute?', 'geh spiel sport', 'GEHEIM sport Veranstaltung', 'sport ist heute', 'sport kostet Geld!']} 
    pred = data, label              
    data_frame = pd.DataFrame(pred)
    return text, data_frame

def fetch_URLSpam(data_home='data'):
    URL_LINGSPAM = 'http://nlp.cs.aueb.gr/software_and_datasets/lingspam_public.tar.gz'
    if not os.path.exists(data_home + '/lingspam_public.tar.gz'):
        urlretrieve(URL_LINGSPAM, data_home + '/lingspam_public.tar.gz')
    df = pd.DataFrame(columns=['text', 'spam?'])
    with tarfile.open(mode="r:gz", name=data_home+'/lingspam_public.tar.gz') as f:
        # We load only the raw texts. 
        folder = 'lingspam_public/bare/'
        files = [name for name in f.getnames() if name.startswith(folder) and name.endswith('.txt')]
        for name in files:
            m = f.extractfile(name)
            df = df.append({'text':str(m.read(), 'utf-8'), 
                            'spam?':1 if 'spmsg' in name else 0}, 
                            ignore_index=True)
    return df   


def create_pipelines_URLSpam():
    stop = ('stop', StopWordRemovalTransformer())
    lemma = ('lemma', LemmatizeTransformer())
    binz = ('binarizer', CountVectorizer())
    we = ('document embedding', DocEmbeddingVectorizer())
    sel = ('fsel', SelectKBest(score_func=mutual_info_classif, k=100))
    clf = ('cls', BernoulliNB()) # Binary features in the original paper. 
    return Pipeline([binz, sel, clf]),   \
           Pipeline([stop, binz, sel, clf]),  \
           Pipeline([lemma, binz, sel, clf]),     \
           Pipeline([stop, lemma, binz, sel, clf]), \
           Pipeline([stop, lemma, we, sel, clf])


def fetch_spambase(data_home='data'):
    URL_SPAMBASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/'

    columns = []
    with urlopen(URL_SPAMBASE + 'spambase.names') as f:
        content = f.readlines()
    for line in content:
        if str(line,'utf-8').startswith(('word_freq', 'char_freq', 'capital_run')):
            columns.append(str(line,'utf-8').split(':')[0]) 
    columns.append('spam?')
    df = pd.read_csv(URL_SPAMBASE + 'spambase.data', header=None)
    df.columns = columns
    return df

def create_pipeline_spambase():
    clf = ('cls', BernoulliNB()) # Has binary and frequencies. 
    return Pipeline([clf])

