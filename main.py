from __future__ import print_function
import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
import PyPDF2
import re
import csv
import pandas as pd
import numpy as np
from io import StringIO
import gensim
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

from whitespace import remove_whitespace
from whitespace import remove_duplicates
from senttowords import sent_to_words
from preprocess import remove_stopwords
from preprocess import lemmatization

myfile = open('evaldocumentV2.pdf', mode = 'rb')
pdf_text = []
pdf_reader = PyPDF2.PdfFileReader(myfile)
for p in range(pdf_reader.numPages):
    page = pdf_reader.getPage(p)
    pdf_text.append(page.extractText())
myfile.close()
list1 = pdf_text
str1 = ''.join(list1)

nlp = spacy.load('en_core_web_sm')
phrasematcher = PhraseMatcher(nlp.vocab)
matcher = Matcher(nlp.vocab)
doc = nlp(str1)

matched_sents = []

def p_sents(phrasematcher, doc, i, pmatches, label = 'MATCH'):
    match_id, start, end = pmatches[i]
    span = doc[start : end]
    sent = span.sent

    if doc.vocab.strings[match_id] == 'computation':
        matched_ents = 'computation'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'data_defination':
        matched_ents = 'data_defination'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'process':
        matched_ents = 'process'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'constraint':
        matched_ents = 'constraint'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'assumption':
        matched_ents = 'assumption'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'model':
        matched_ents = 'model'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'performance':
        matched_ents = 'performance'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'hardware':
        matched_ents = 'hardware'
        matched_sents.append({'Pattern' : matched_ents, 'Requirements' : sent.text})

def m_sents(matcher, doc, i, matches, label = 'MATCH'):
    match_id, start, end =  matches[i]
    span = doc[start : end]
    sent = span.sent

    if doc.vocab.strings[match_id] == 'Pattern1':
        match_ents = 'Pattern1'
        matched_sents.append({'Pattern' : match_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'Pattern2':
        match_ents = 'Pattern2'
        matched_sents.append({'Pattern' : match_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'Pattern3':
        match_ents = 'Pattern3'
        matched_sents.append({'Pattern' : match_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'Pattern4':
        match_ents = 'Pattern4'
        matched_sents.append({'Pattern' : match_ents, 'Requirements' : sent.text})
    elif doc.vocab.strings[match_id] == 'Pattern5':
        match_ents = 'Pattern5'
        matched_sents.append({'Pattern' : match_ents, 'Requirements' : sent.text})

computation = ['method', 'technique', 'approach', 'algorithm']
data_defination = ['data','information','file','format']
process = ['calculate','compute','discretize', 'input', 'output']
constraint = ['constraint','restriction', 'restraint', 'limitation']
assumption = ['assume','assumption', 'hypothesis']
model = ['model','framework']
performance = ['efficient','speed', 'robust']
hardware = ['CPU','memory']

phrasematcher.add('computation', p_sents, *[nlp(text) for text in computation])
phrasematcher.add('data_defination', p_sents, *[nlp(text) for text in data_defination])
phrasematcher.add('process', p_sents, *[nlp(text) for text in process])
phrasematcher.add('constraint', p_sents, *[nlp(text) for text in constraint])
phrasematcher.add('assumption', p_sents, *[nlp(text) for text in assumption])
phrasematcher.add('model', p_sents, *[nlp(text) for text in model])
phrasematcher.add('performance', p_sents, *[nlp(text) for text in performance])
phrasematcher.add('hardware', p_sents, *[nlp(text) for text in hardware])


pattern1 = [{'POS': 'NOUN'}, {'POS': 'VERB', 'TAG' : 'MD'}]
pattern2 = [{'POS': 'NOUN'}, {'POS': 'VERB', 'TAG' : 'VBZ'}]
pattern3 = [{'POS' : 'PROPN', 'TAG' : 'NNP'}, {'POS': 'VERB', 'TAG' : 'MD'}]
pattern4 = [{'POS' : 'PROPN'}, {'POS': 'VERB', 'TAG' : 'MD'}]
pattern5 = [{'POS' : 'NOUN', 'TAG' : 'NN'},{'POS' : 'NOUN', 'TAG' : 'NNS'} ,{'POS': 'VERB', 'TAG' : 'MD'}]

matcher.add('Pattern1', m_sents, pattern1)  
matcher.add('Pattern2', m_sents, pattern2)
matcher.add('Pattern3', m_sents, pattern3)
matcher.add('Pattern4', m_sents, pattern4)
matcher.add('Pattern5', m_sents, pattern5)

pmatches = phrasematcher(doc)
matches = matcher(doc)

df = pd.DataFrame(matched_sents)

df1 = remove_whitespace(df)
df2 = remove_duplicates(df1)
df3 = df2.reset_index()
data = df3.Requirements.values.tolist()

data_words = list(sent_to_words(data))

bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 100)
trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
     return[bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return[trigram_mod[bigram_mod[doc]]for doc in texts]

data_words_nostops = remove_stopwords(data_words)
data_words_bigram = make_bigrams(data_words_nostops)
nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigram, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.dictionary.Dictionary(data_lemmatized)
texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

def get_model_list(dictionary, corpus, texts, limit, start=2, step=3):
    
    model_list = []
    for num_topics in range(start, limit, step):
        
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)

    return model_list
model_list = get_model_list(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=3, limit=18, step=3)
optimal_model = model_list[0]
model_topics = optimal_model.show_topics(formatted=False)

def topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    
    sent_topics_df = pd.DataFrame()
    
    for i, row in enumerate(ldamodel[corpus]):

        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1) 
    sent_topics_df['Pattern'] = pd.Series(df3['Pattern'])
    return(sent_topics_df)

df_topic_sents_keywords = topics_sentences(ldamodel = optimal_model, corpus = corpus, texts = data)
df_topic_sents_keywords.columns = ['Topic', 'Topic Keywords', 'Requirements Candidates', 'Pattern']
print(df_topic_sents_keywords)


