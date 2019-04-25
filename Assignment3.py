from string import punctuation
import nltk
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#glove
glovebath = 'glove.6B.50d.txt'


with open(glovebath, 'r', encoding="utf8") as f:
        word_vocab = set()
        word2vector = {}
        for line in f:
            line_ = line.strip()
            words_Vec = line_.split()
            #print(words_Vec[1])
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:], dtype=float)
print("Total Words in DataSet:", len(word_vocab))

TEXT_DATA_DIR = 'txt_sentoken'
texts = []
labels_index = {}
labels = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
                fpath = os.path.join(path, fname)
                f = open(fpath, "r").read()
                texts.append(f)
                labels.append((label_id))

print('Labels = ', len(labels))
print('Classes are:')
for key in labels_index:
    print(key)

wordsOfDoc = []
withoutpunc = []
map = []
map2 = []
dictionary = {}
dicvalues= []


def removepunc(text):
    result = ""
    for i in text:
        if i not in punctuation:
            result += i
    return result


for i in range(len(texts)):
    x = removepunc(texts[i])
    withoutpunc.append(x)


for i in range(len(withoutpunc)):
    word = nltk.word_tokenize(withoutpunc[i])
    wordsOfDoc.append(word)
    for j in range(len(word)):
        if word[j] in word2vector:
            dictionary[withoutpunc[i]] = np.sum(word2vector[word[j]])


for i in dictionary.values():
    dicvalues.append((i))

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
