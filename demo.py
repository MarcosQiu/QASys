from paraRetrival_bigram import *
from gensim.models import Word2Vec
from process import *
from model import *
from util import *
import numpy as np
import json
import pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# paras and questions are just list of lemmas
all_paras, all_questions, para_max_len, question_max_len = collect_all_para_question()

corpus = ['<\s>']
for sents in [all_paras, all_questions]:
    for sent in sents:
        corpus += sent
token_to_int_mapping = vocabulary_dict(corpus)

doc = json.load(open('documents.json'))
train_para = list()
train_question = list()
train_label_start = list()
train_label_end = list()

train_items = json.load(open('training.json'))
for item in train_items:
    pairs = get_index_pair(item)
    if pairs is not None:
        start_index, end_index = pairs
        paras = [[lmz(token) for token in nltk.word_tokenize(doc[item['docid']]['text'][item['answer_paragraph']])]]
        questions = [[lmz(token) for token in nltk.word_tokenize(item['question'])]]
        padding(paras, para_max_len)
        padding(questions, question_max_len)
        para = paras[0]
        question = questions[0]
        train_para.append(int_encoded_text(para, token_to_int_mapping))
        train_question.append(int_encoded_text(question, token_to_int_mapping))
        train_label_start.append(to_one_hot(start_index, para_max_len))
        train_label_end.append(to_one_hot(end_index, para_max_len))
padding(all_paras, para_max_len)
padding(all_questions, question_max_len)

w2v = Word2Vec(all_paras + all_questions, size = 300, min_count = 1)
weights = [None] * len(token_to_int_mapping)
for key in token_to_int_mapping:
    weights[token_to_int_mapping[key]] = w2v.wv[key]

para_question_len = [para_max_len, question_max_len]
weights = np.array(weights)
train_question = np.array(train_question)
train_para = np.array(train_para)
train_label_start = np.array(train_label_start)
train_label_end = np.array(train_label_end)

print('preprocessing done...')
save_obj(token_to_int_mapping, 'token_to_int_mapping')
save_obj(para_question_len, 'para_question_len')
np.savez('train_data.npz', weights = np.array(weights), train_question = np.array(train_question),
           train_para = np.array(train_para), train_label_start = np.array(train_label_start),
           train_label_end = np.array(train_label_end))
print('data saved!')


# model = lstm_model(token_to_int_mapping, np.array(weights), para_max_len, question_max_len)
# print('training...')
# model.train(np.array(train_question), np.array(train_para), np.array(train_label_start))


# pred_start_model = lstm_model(all_paras, all_questions, para_max_len, question_max_len, 'start',w2v)
# pred_start_model.train()
# train = json.load(open('training.json'))
# train_x, train_y = prepare_data(train[:2], para_max_len, question_max_len, 'end', w2v)

# print (train_x.shape)
# print (train_y.shape)
# print (train_y)
