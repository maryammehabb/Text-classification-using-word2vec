from string import punctuation
import nltk
import os
import sys
import numpy as np
import pandas as pd

#glove
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

glovebath = 'glove.6B.50d.txt'

#def read_data(file_name):
with open(glovebath,'r', encoding="utf8") as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            #print(words_Vec[1])
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
print("Total Words in DataSet:",len(word_vocab))

TEXT_DATA_DIR = 'txt_sentoken'
texts = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
labels = [] # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    print(name)
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                f = open(fpath, "r").read()
                texts.append(f)
                labels.append((label_id))

'''print('Found %s texts.' % len(texts))
print('nLabels = ', len(labels))
print('Classes are:\n ')
for key in labels_index:
    print(key)
print(labels_index)'''

wordsOfDoc = []
withoutpunc = []


def removepunc(text):
    result = ""
    for i in text:
        if i not in punctuation:
            result += i
    return result


for i in range(len(texts)):
    x = removepunc(texts[i])
    withoutpunc.append(x)
print("Words without punctuation: ")
print(len(withoutpunc))
map = []
map2 = []
dictionary = {}
dicvalues= []
for i in range(len(withoutpunc)):
    word = nltk.word_tokenize(withoutpunc[i])
    wordsOfDoc.append(word)
    for j in range(len(word)):
        if word[j] in word2vector:
            dictionary[withoutpunc[i]] = np.sum(word2vector[word[j]])

print(len(dictionary.values()))
print(dictionary.values())
for i in dictionary.values():
    dicvalues.append((i))

print(len(dicvalues))
print(dicvalues)
print(labels)
print(len(labels))
'''print(texts)
print(len(texts))
print(labels)
print(len(labels))'''
dicvalues = ([dicvalues])
labels = ([labels])
print(labels)
print(dicvalues)
X_train, X_test, y_train, y_test = train_test_split(dicvalues, labels, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
yPrediction = model.predict(X_test)
print(yPrediction)
x =accuracy_score(y_test, yPrediction)
print(x)
a = x * 100
print(a)
