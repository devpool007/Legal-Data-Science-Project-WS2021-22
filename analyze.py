import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
from joblib import dump, load
from sbd_utils import text2sentences
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.language import Language
from spacy.symbols import ORTH
import spacy
import re
from pathlib import Path
import sys

nlp = spacy.load("en_core_web_sm")
def u_tokenize(txt):
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    for t in tokens:
        if t.pos_ == 'PUNCT':
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        else:
            clean_tokens.append(t.lemma_)
    #remove all non alphanumerics
    clean_tokens = [re.sub(r'\W', '', t).lower() 
                    for t in clean_tokens]
    for i in clean_tokens:
        if i == '':
            clean_tokens.remove(i)
    return clean_tokens

def run(filepath):
	
	spans = []
	spans_txt = []

	file = open(filepath,'r')
	txt = file.read() 
	doc = text2sentences(txt,offsets=True)
	doc_txt = text2sentences(txt,offsets=False)
	for j in doc:
		spans.append({'start': j[0] , 'end': j[1], 'start_normalized': j[0] / len(doc)})

	for j in doc_txt:
		spans_txt.append({'txt' : j})
	file.close()
	spacy_pacy = load('spacy_tfidf_vectorizer.joblib')
	tfidf = spacy_pacy.transform([s['txt'] for s in spans_txt]).toarray()
	starts_normalized = np.array([s['start_normalized'] for s in spans])
	X = np.concatenate((tfidf, np.expand_dims(starts_normalized, axis=1)), axis=1)
	clf = load('RF_TFIDF.joblib') 
	ans = clf.predict(X)

	return ans,doc_txt

if __name__ == '__main__':
	print("Program is now running...\n")
	filepath = sys.argv[1]
	predictions,txt = run(filepath)
	print("SENTENCE :===> TYPE PREDICTED")
	print("=========================\n")
	n = 0
	for i in txt:
		print("{}:===>{}".format(i,predictions[n]))
		print("=========================\n")
		n = n + 1
	