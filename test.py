# # exactly match features
# from process import *
# from util import *
# import numpy as np
# import json
# import nltk
# import pickle
#
# def load_obj(name):
#     with open(name + '.pkl', 'rb') as f:
#         return pickle.load(f)
#
# para_question_len = load_obj('para_question_len')
# para_max_len = para_question_len[0]
# data_file = np.load('train_data.npz')
# train_question = data_file['train_question']
# num_train_instance = train_question.shape[0]
#
# train = json.load(open('training.json'))
# doc = json.load(open('documents.json'))
#
# exact_match = np.zeros((num_train_instance, para_max_len, 3))
# instance_index = 0
# for train_instance in train_items:
# 	if get_index_pair(train_instance, doc) is None:
# 		continue
# 	question_token = nltk.word_tokenize(train_instance['question'])
# 	question_lower = [token.lower() for token in question_token]
# 	question_lemma = [lmz(token) for token in question_token]
#
# 	para_token = nltk.word_tokenize(doc[train_instance['docid']]['text'][train_instance['answer_paragraph']])
# 	for token_index in range(len(para_token)):
# 		if para_token[token_index] in question_token:
# 			exact_match[instance_index, token_index, 0] = 1
# 		if para_token[token_index].lower() in question_lower:
# 			exact_match[instance_index, token_index, 1] = 1
# 		if lmz(para_token[token_index]) in question_lemma:
# 			exact_match[instance_index, token_index, 2] = 1
# 	instance_index += 1
#
# print(exact_match)
# np.save('exact_match.npy', exact_match)


from paraRetrival_bigram import *
from prepare_data import *
from train import *
from pred import *

main_process(fileName = 'testing.json')
prepare_data()
train()
predict()
