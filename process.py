from gensim.models import Word2Vec
from paraRetrival_bigram import *
import nltk
from util import *
import numpy as np
import json

def vocabulary_dict(corpus):
    result = dict()
    for token in corpus:
        result.setdefault(token, len(result))
    return result

def int_encoded_text(text, dict_to_int):
    return [dict_to_int.get(token, dict_to_int['<\s>']) for token in text]

def collect_all_para_question():
    doc = json.load(open('documents.json'))
    train = json.load(open('training.json'))
    dev = json.load(open('devel.json'))
    test = json.load(open('testing.json'))

    # get all the questions
    all_questions = list()
    for item in train:
        all_questions.append([lmz(token) for token in nltk.word_tokenize(item['question'])])

    # get all the documents
    all_paras = list()
    for item in doc:
        for para in item['text']:
            all_paras.append([lmz(token) for token in nltk.word_tokenize(para)])

    para_max_len = np.max([len(para) for para in all_paras])
    question_max_len = np.max([len(question) for question in all_questions])

    # padding(all_paras, para_max_len)
    # padding(all_questions, question_max_len)

    return all_paras, all_questions, para_max_len, question_max_len

def padding(items, max_len):
    for item in items:
        item += (['<\s>'] * (max_len - len(item)))

def to_one_hot(class_num, total_size):
    result = [0] * total_size
    result[class_num] = 1
    return np.array(result)

def get_index_pair(item):
    para_id = item['answer_paragraph']
    doc_id = item['docid']
    doc = json.load(open('documents.json'))
    answer = [lmz(token) for token in nltk.word_tokenize(item['text'])]
    para = [lmz(token) for token in nltk.word_tokenize(doc[doc_id]['text'][para_id])]

    for index_para in range(len(para) - len(answer) + 1):
        flag = True
        for index_answer in range(len(answer)):
            if para[index_para + index_answer] != answer[index_answer]:
                flag = False
                break
        if flag:
            return index_para, index_para + len(answer) - 1
        else:
            continue
    return None

def prepare_data(train_this_batch, para_max_len, question_max_len, start_end, w2v):
    train_x = None
    train_y = list()
    doc = json.load(open('documents.json'))
    valid_item = 0
    for train_item in train_this_batch:
        pairs = get_index_pair(train_item)
        if pairs is None:
            continue
        valid_item += 1
        label = pairs[0]
        if start_end == 'end':
            label = pairs[1]
        train_y.append(to_one_hot(label, para_max_len))
        para = [lmz(token) for token in nltk.word_tokenize(doc[train_item['docid']]['text'][train_item['answer_paragraph']])]
        question = [lmz(token) for token in nltk.word_tokenize(train_item['question'])]
        padding([para], para_max_len)
        padding([question], question_max_len)
        vec = None
        for collections in [para, question]:
            for word in collections:
                if vec is None:
                    vec = np.array(w2v.wv[word]).reshape(1, 300)
                else:
                    vec = np.vstack((vec, np.array(w2v.wv[word]).reshape(1,300)))
        if train_x is None:
            train_x = vec.reshape(1, para_max_len + question_max_len, 300)
        else:
            train_x = np.append(train_x, vec.reshape(1, para_max_len + question_max_len, 300), axis = 0)
    train_y = np.array(train_y).reshape(valid_item, para_max_len)

    return train_x, train_y



if __name__ == '__main__':
    count = 0
    for item in json.load(open('training.json')):
        res = get_index_pair(item)
        if res is None:
            count += 1
    print(count)
