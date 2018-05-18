import numpy as np
from util import *
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# define the lemmatize function
def lmz(word):
    lemmatizer = WordNetLemmatizer()
    word = word.lower()
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma

# get bag-of-words from tokenized document
def get_BOW(text):
    BOW = dict()
    stop_words = set(stopwords.words('english'))
    for word in text:
        word_lemma = lmz(word)
        if word_lemma in stop_words:
            continue
        BOW[word_lemma] = BOW.get(word_lemma,0) + 1
    return BOW

def get_bigram(token_list):
    result = list()
    result.append('_' + token_list[0])
    for i in range(len(token_list) - 1):
        result.append(token_list[i] + '_' + token_list[i + 1])
    result.append(token_list[-1] + '_')
    return result + token_list

# function for deriving count given sense
def get_count(sense, origin_word):
    for lemma in sense.lemmas():
        if lemma.name() == origin_word:
            return lemma.count()
    return 0

# function for deciding if a word is ambiguous or not
def not_ambiguous(word):
    syn_sets = wn.synsets(word)
    if len(syn_sets) < 2:
        return True
    count_1 = 0
    count_2 = 0
    for synset in syn_sets:
        count = get_count(synset, word)
        if count > count_2:
            if count > count_1:
                count_2 = count_1
                count_1 = count
            else:
                count_2 = count
    if count_1 >= count_2 * 5:
        return True
    else:
        return False
