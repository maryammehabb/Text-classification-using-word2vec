import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model

print("hai")
TEXT_DATA_DIR = 'D:\Mariam\/4th year\/2nd term\/NLP\Assignments\/New folder\/txt_sentoken'
texts = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
labels = [] # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    print(name)
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        print("mm")
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            #if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n') # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)
print('Found %s texts.' % len(texts))
print('nLabels = ', len(labels))
print('Classes are:\n ')
for key in labels_index:
    print(key)

'''MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index # the dictionary
print('Found %s unique tokens.' % len(word_index)) #only top MAX_NUM_WORDS will be used to generate the sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of samples:', data.shape)
print('Sampele:(the zeros at the begining are for padding text to max length)')
print(data[2])

labels_matrix = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels_matrix.shape)
print('Sample label:\n',labels_matrix[1590])'''