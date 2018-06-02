import json
import nltk
import numpy as np
from util import *
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

'''
This file is used to retrive relative paragraghs from the article.
Using TF-IDF weights, and inverted indexes to speed up the algorithm.
'''

def main_process(fileName = 'devel.json',testing = False):

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
        train = json.load(open(fileName))
        success = 0
        length = 0
        for train_item in train:
            bigram_list = get_bigram(nltk.word_tokenize(train_item['question']))
            question = get_BOW(bigram_list)
            vectorizer, transformer = transformer_pair[train_item['docid']]
            score = [0] * len(doc_dict[train_item['docid']])
            word_feature_map = vectorizer.vocabulary_
            for word_from_question in question:
                for index in range(len(doc_dict[train_item['docid']])):
                    if word_from_question in doc_dict[train_item['docid']][index]:
                        score[index] += (tf_idf_matrix_reg_doc[train_item['docid']][index, word_feature_map[word_from_question]])
            if train_item['answer_paragraph'] in np.argsort(score)[-2:]:
                success += 1
            length += 1
        print('The retrival accuracy is', success * 1.0 / float(length))
    else:
        testing = json.load(open(fileName))
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
            related_para[test_item['id']] = np.argsort(score)[-2:]
        save_obj(related_para, 'related_para_top_2')
        return related_para


if __name__ == '__main__':
    main_process(testing = True)
