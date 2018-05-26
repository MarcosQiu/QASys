import numpy as np
from model import *
from util import *

def train():
    exact_match = np.load('exact_match.npy')
    token_to_int_mapping = load_obj('token_to_int_mapping')
    para_question_len = load_obj('para_question_len')
    tag_to_int_mapping = load_obj('tag_to_int_mapping')
    para_max_len = para_question_len[0]
    question_max_len = para_question_len[1]
    data_file = np.load('train_data.npz')
    weights = data_file['weights']
    train_question = data_file['train_question']
    train_para = data_file['train_para']
    train_label_start = data_file['train_label_start']
    train_label_end = data_file['train_label_end']
    train_pos_para = data_file['train_pos_para']
    train_pos_question = data_file['train_pos_question']
    exact_match = data_file['exact_match']

    model = lstm_model(token_to_int_mapping, np.array(weights), para_max_len, question_max_len, tag_to_int_mapping)
    print('training...')
    model.train(train_question, train_para, train_pos_question, train_pos_para, exact_match, train_label_start, 'best_model_start_bilstm.model')
    model.train(train_question, train_para, train_pos_question, train_pos_para, exact_match, train_label_end, 'best_model_end_bilstm.model')

if __name__ == '__main__':
    train()
