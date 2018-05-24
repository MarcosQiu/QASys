import pickle
from paraRetrival_bigram import *

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

related_para = main_process()
save_obj(related_para, 'related_para')
