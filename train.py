import numpy as np
import pickle

def load_model(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

token_to_int_mapping = load_model('token_to_int_mapping')
para_question_len = load_model('para_question_len')
para_max_len = para_question_len[0]
question_max_len = para_question_len[1]
data_file = np.load('train_data.npz')
weights = data_file['weights']
train_question = data_file['train_question']
train_para = data_file['train_para']
train_label_start = data_file['train_label_start']
train_label_end = data_file['train_label_end']

model = lstm_model(token_to_int_mapping, np.array(weights), para_max_len, question_max_len)
print('training...')
model.train(np.array(train_question), np.array(train_para), np.array(train_label_start))
