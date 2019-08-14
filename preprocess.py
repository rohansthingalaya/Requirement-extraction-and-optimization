import spacy
import gensim
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

nlp = spacy.load('en_core_web_sm')

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
def remove_stopwords(texts):
    token = [[word for word in simple_preprocess(str(doc)) if word not in spacy_stopwords] for doc in texts]
    return token
    
def lemmatization(texts, allowed_postags = ['Noun', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out