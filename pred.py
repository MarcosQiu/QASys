from keras.models import load_model
import numpy as np
from process import *
from util import *
import nltk
import json
import pickle
import io

def predict():
    with io.open('answer.txt', "w", encoding="utf-8") as answer_file:

        top_k = 2
        start_model = load_model('best_model_start_bilstm.model')
        end_model = load_model('best_model_end_bilstm.model')

        doc = json.load(open('documents.json'))
        testing = json.load(open('testing.json'))
        answer_file.write('id,answer\n')

        para_question_len = load_obj('para_question_len')
        related_para = load_obj('related_para_top_2')
        tag_to_int_mapping = load_obj('tag_to_int_mapping')
        print(tag_to_int_mapping)
        token_to_int_mapping = load_obj('token_to_int_mapping')
        para_max_len = para_question_len[0]
        question_max_len = para_question_len[1]

        inputs_question = list()
        inputs_para = list()
        inputs_question_pos = list()
        inputs_para_pos = list()

        for index in range(len(related_para)):
            docID = testing[index]['docid']
            question = testing[index]['question']
            question = [[lmz(token) for token in nltk.word_tokenize(question)]]
            padding(question, question_max_len)
            question = question[0]
            int_question = [token_to_int_mapping.get(lemma, token_to_int_mapping['<\s>']) for lemma in question]
            question_pos = pos_sequence(question)
            int_question_pos = [tag_to_int_mapping.get(pos_tag, 0) for pos_tag in question_pos]

            inputs_question.append(int_question)
            inputs_question_pos.append(int_question_pos)

            cur_related_para = [doc[docID]['text'][para_index] for para_index in related_para[index]]
            cur_related_para_token = list()
            for para in cur_related_para:
                cur_related_para_token.append([lmz(token) for token in nltk.word_tokenize(para)])
            padding(cur_related_para_token, para_max_len)
            cur_related_para_pos = [pos_sequence(para) for para in cur_related_para_token]

            int_related_para = list()
            int_related_para_pos = list()
            for para in cur_related_para_token:
                int_related_para.append([token_to_int_mapping.get(lemma, token_to_int_mapping['<\s>']) for lemma in para])
            for para in cur_related_para_pos:
                int_related_para_pos.append([tag_to_int_mapping.get(pos_tag, 0) for pos_tag in para])
            inputs_para.append(int_related_para)
            inputs_para_pos.append(int_related_para_pos)

        for index in range(len(inputs_question)):
            question = inputs_question[index]
            question_pos = inputs_question_pos[index]
            paras = inputs_para[index]
            paras_pos = inputs_para_pos[index]

            question_array = None
            question_pos_array = None
            para_array = None
            para_pos_array = None
            exact_match = np.zeros((top_k, para_max_len, 3))

            docID = testing[index]['docid']
            text = doc[docID]['text']

            question_origin = nltk.word_tokenize(testing[index]['question'])
            question_lower = [token.lower() for token in question_origin]
            question_lemma = [lmz(token) for token in question_origin]
            cur_related_para = related_para[index]

            for para_index in range(top_k):
                if para_index == 0:
                    question_array = np.array(question)
                    question_pos_array = np.array(question_pos)
                    para_array = np.array(paras[para_index])
                    para_pos_array = np.array(paras_pos[para_index])
                else:
                    question_array = np.vstack((question_array, question))
                    question_pos_array = np.vstack((question_pos_array, question_pos))
                    para_array = np.vstack((para_array, paras[para_index]))
                    para_pos_array = np.vstack((para_pos_array, paras_pos[para_index]))

                # preparing for exact match features
                para_origin = nltk.word_tokenize(text[cur_related_para[para_index]])

                for token_index in range(len(para_origin)):
                	if para_origin[token_index] in question_origin:
                		exact_match[para_index, token_index, 0] = 1
                	if para_origin[token_index].lower() in question_lower:
                		exact_match[para_index, token_index, 1] = 1
                	if lmz(para_origin[token_index]) in question_lemma:
                		exact_match[para_index, token_index, 2] = 1


            start = start_model.predict([question_array, para_array, question_pos_array, para_pos_array, exact_match])
            end = end_model.predict([question_array, para_array, question_pos_array, para_pos_array, exact_match])

            overall_prob = list()
            overall_optimal_pair = list()

            for para_index in range(top_k):
                pred_start = start[para_index]
                pred_end = end[para_index]
                max_prob_this_run = 0
                optimal_pair_this_run = None

                for start_index in range(para_max_len):
                    optimal_index_pair_fix_start = None
                    max_prob_fix_start = 0
                    for end_index in range(start_index, min(start_index + 15, para_max_len)):
                        if pred_start[start_index] * pred_end[end_index] > max_prob_fix_start:
                            max_prob_fix_start = pred_start[start_index] * pred_end[end_index]
                            optimal_index_pair_fix_start = (start_index, end_index)
                    if max_prob_fix_start > max_prob_this_run:
                        max_prob_this_run = max_prob_fix_start
                        optimal_pair_this_run = optimal_index_pair_fix_start
                overall_prob.append(max_prob_this_run)
                overall_optimal_pair.append(optimal_pair_this_run)

            win_para = related_para[index][np.argmax(overall_prob)]
            to_start, to_end = overall_optimal_pair[np.argmax(overall_prob)]
            text = [nltk.word_tokenize(doc[testing[index]['docid']]['text'][win_para])]
            padding(text, para_max_len)
            text = text[0]
            answer_token = text[to_start: to_end + 1]
            answer = ''
            for token in answer_token:
                if answer == '':
                    answer = token
                else:
                    answer += (' ' + token)

            answer_file.write(str(index) + ',' + answer + '\n')

            if (index + 1) % 100 == 0:
                print(str(index + 1) + ' answers generalted...')

        answer_file.close()

if __name__ == '__main__':
    predict()

# start_model.predict([inputs_question, inputs_para, inputs_question_pos, inputs_para_pos])
# end_model.predict([inputs_question, inputs_para, inputs_question_pos, inputs_para_pos])
