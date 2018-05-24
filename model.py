from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras
from process import *
import os
import numpy as np
import nltk
import json

class lstm_model(object):

    def __init__(self, token_to_int_mapping, weights, para_max_len, question_max_len, tag_to_int_mapping):
        self.token_to_int_mapping = token_to_int_mapping
        self.weights = weights
        self.tag_to_int_mapping = tag_to_int_mapping
        # self.all_paras = all_paras
        # self.all_questions = all_questions
        self.para_max_len = para_max_len
        self.question_max_len = question_max_len
        self.bedding_dim = 300
        self.batch_size = 128
        self.epoches = 10
        print('tag_to_int_mapping\n', tag_to_int_mapping)
        inputs_exact_match = Input(shape = (self.para_max_len, 3))

        inputs_question_pos = Input(shape = (self.question_max_len,))
        embedded_question_pos = Embedding(input_dim = len(self.tag_to_int_mapping),
                                          output_dim = len(self.tag_to_int_mapping),
                                          input_length = self.question_max_len,
                                          weights = [np.eye(len(self.tag_to_int_mapping))])(inputs_question_pos)
        # lstm_question_pos = LSTM(256)(embedded_question_pos)

        inputs_question = Input(shape = (self.question_max_len,))
        embedded_question = Embedding(input_dim = len(self.token_to_int_mapping),
                                      output_dim = self.bedding_dim,
                                      input_length = self.question_max_len,
                                      weights = [self.weights])(inputs_question)


        inputs_para_pos = Input(shape = (self.para_max_len,))
        embedded_para_pos = Embedding(input_dim = len(self.tag_to_int_mapping),
                                          output_dim = len(self.tag_to_int_mapping),
                                          input_length = self.para_max_len,
                                          weights = [np.eye(len(self.tag_to_int_mapping))])(inputs_para_pos)
        # lstm_para_pos = LSTM(256)(embedded_para_pos)


        inputs_para = Input(shape = (self.para_max_len,))
        embedded_para = Embedding(input_dim = len(self.token_to_int_mapping),
                                  output_dim = self.bedding_dim,
                                  input_length = self.para_max_len,
                                  weights = [self.weights])(inputs_para)

        merged_embedded_para = keras.layers.concatenate([embedded_para, embedded_para_pos, inputs_exact_match])
        merged_embedded_question = keras.layers.concatenate([embedded_question, embedded_question_pos])

        lstm_para = LSTM(64, return_sequences = True)(merged_embedded_para)
        lstm_question = LSTM(64, return_sequences = True)(merged_embedded_question)
        lstm_para = Flatten()(lstm_para)
        lstm_question = Flatten()(lstm_question)
        merged = keras.layers.concatenate([lstm_para, lstm_question])
        output = Dense(self.para_max_len, activation = 'softmax')(merged)

        self.model = Model(inputs = [inputs_question, inputs_para, inputs_question_pos, inputs_para_pos, inputs_exact_match], outputs = output)
        self.model.compile(optimizer = 'adam',
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])
        print(self.model.summary())

        # self.model = Sequential()
        # self.model.add(LSTM(1, input_shape=self.input_size, return_sequences = False))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(self.para_max_len, activation = 'softmax'))
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, train_question, train_para, train_question_pos, train_para_pos, inputs_exact_match, train_lables):
        print(train_question.shape, train_para.shape, train_question_pos.shape, train_para_pos.shape, train_lables.shape)
        best_model = 'best_model_end.model'
        checkpoint = ModelCheckpoint(filepath = best_model, verbose = 1, save_best_only = True)
        self.model.fit([train_question, train_para, train_question_pos, train_para_pos, inputs_exact_match], train_lables, batch_size = self.batch_size, epochs = self.epoches, validation_split = 0.1, callbacks = [checkpoint])
