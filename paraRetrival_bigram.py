import json
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

'''
This file is used to retrive relative paragraghs from the article.
Using TF-IDF weights, and inverted indexes to speed up the algorithm.
'''

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
    for word in text:
        word_lemma = lmz(word)
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

def main_process(testing = False):
    '''
    load documents from json
    [
        {
            docid: num
            text: [
                    para_1,
                    para_2,
                    ...
                ]
        },
        ...
    ]
    '''
    doc = json.load(open('documents.json'))

    # the bag-of-words representations for each article
    doc_dict = [None] * len(doc)
    for doc_item in doc:
        para_dict = list()
        for para in doc_item['text']:
            # get the bigram set
            bigram_list = get_bigram(nltk.word_tokenize(para))
            para_dict.append(get_BOW(bigram_list))
        doc_dict[doc_item['docid']] = para_dict

    # For each document, get the TF-IDF weight, store all
    # the information for later use
    transformer_pair = [None] * len(doc_dict)
    tf_idf_matrix_reg_doc = [None] * len(doc_dict)

    for doc_index in range(len(doc_dict)):
        vectorizer = DictVectorizer()
        transformer = TfidfTransformer()

        term_para_matrix = vectorizer.fit_transform(doc_dict[doc_index])
        tf_idf_matrix_reg_doc[doc_index] = transformer.fit_transform(term_para_matrix)
        transformer_pair[doc_index] = (vectorizer, transformer)

    if testing:
        train = json.load(open('training.json'))
        success = 0
        length = 0
        for train_item in train:
            bigram_list = get_bigram(nltk.word_tokenize(train_item['question']))
            question = get_BOW(bigram_list)
            vectorizer, transformer = transformer_pair[train_item['docid']]
            score = [0] * len(doc_dict[train_item['docid']])
            word_feature_map = vectorizer.vocabulary_
            for word_from_question in question:
                flag = True
                if not not_ambiguous(word_from_question):
                    flag = False
                for index in range(len(doc_dict[train_item['docid']])):
                    if word_from_question in doc_dict[train_item['docid']][index]:
                        if flag:
                            score[index] += (tf_idf_matrix_reg_doc[train_item['docid']][index, word_feature_map[word_from_question]])
                        else:
                            score[index] += (tf_idf_matrix_reg_doc[train_item['docid']][index, word_feature_map[word_from_question]] * 2.0 / 1.6)
            if train_item['answer_paragraph'] in np.argsort(score)[-6:]:
                success += 1
            length += 1
        print success * 1.0 / float(length)
    else:
        testing = json.load(open('testing.json'))
        related_para = [None] * len(testing)

        for test_item in testing:
            question = get_BOW(nltk.word_tokenize(test_item['question']))
            vectorizer, transformer = transformer_pair[test_item['docid']]
            score = [0] * len(doc_dict[test_item['docid']])
            word_feature_map = vectorizer.vocabulary_
            for word_from_question in question:
                for index in range(len(doc_dict[test_item['docid']])):
                    if word_from_question in doc_dict[test_item['docid']][index]:
                        score[index] += tf_idf_matrix_reg_doc[test_item['docid']][index, word_feature_map[word_from_question]]
            related_para[test_item['id']] = np.argsort(score)[-5:]
        return related_para


if __name__ == '__main__':
    main_process(testing = True)
