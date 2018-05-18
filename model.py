from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import load_model
import keras
from process import *
import os
import numpy as np
import nltk
import json

class lstm_model(object):

    def __init__(self, token_to_int_mapping, weights, para_max_len, question_max_len):
        self.token_to_int_mapping = token_to_int_mapping
        self.weights = weights
        # self.all_paras = all_paras
        # self.all_questions = all_questions
        self.para_max_len = para_max_len
        self.question_max_len = question_max_len
        self.bedding_dim = 300
        self.batch_size = 128
        self.epoches = 10

        inputs_question = Input(shape = (self.question_max_len,))
        embedded_question = Embedding(input_dim = len(self.token_to_int_mapping),
                                      output_dim = self.bedding_dim,
                                      input_length = self.question_max_len,
                                      weights = [self.weights])(inputs_question)
        lstm_question = LSTM(256)(embedded_question)


        inputs_para = Input(shape = (self.para_max_len,))
        embedded_para = Embedding(input_dim = len(self.token_to_int_mapping),
                                  output_dim = self.bedding_dim,
                                  input_length = self.para_max_len,
                                  weights = [self.weights])(inputs_para)
        lstm_para = LSTM(256)(embedded_para)
        merged = keras.layers.concatenate([lstm_question, lstm_para])
        output = Dense(self.para_max_len, activation = 'softmax')(merged)

        self.model = Model(inputs = [inputs_question, inputs_para], outputs = output)
        self.model.compile(optimizer = 'adam',
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])

        # self.model = Sequential()
        # self.model.add(LSTM(1, input_shape=self.input_size, return_sequences = False))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(self.para_max_len, activation = 'softmax'))
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, train_question, train_para, train_lables):
        self.model.fit([train_question, train_para], train_lables, batch_size = self.batch_size, epochs = self.epoches, validation_split = 0.1)
