import nltk
import os
import numpy as np
from string import punctuation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#glove
glovebath = 'glove.6B.50d.txt'
with open(glovebath, 'r', encoding="utf8") as f:
        words = set()
        word2vector = {}
        for line in f:
            line_ = line.strip()
            words_Vec = line_.split()
            words.add(words_Vec[0])
            word2vector[words_Vec[0]] = list(map(float, words_Vec[1:]))
print("Total Words in DataSet:", len(words))

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
                texts.append([f])
                labels.append((label_id))
print('Classes are:')
for key in labels_index:
    print(key)
withoutpunc = []


def removepunc(text):
    result = ""
    for i in text:
        if i not in punctuation:
            result += i
    return result


for i in range(len(texts)):
    x = removepunc(texts[i])
    withoutpunc.append([x])


def emmbedding(type, method):
    result = []
    wordsOfDoc = []
    for i in range(len(type)):
        #print(type[i])
        tmp = []
        for j in type[i]:
            #print(j)
            word = nltk.word_tokenize(j)
            wordsOfDoc.append(word)
            for p in word:
                if p in word2vector:
                    tmp.append(word2vector[p])
                else:
                    tmp.append([0]*50)
            r = [sum(i) if method == "SUM" else sum(i)/len(i) for i in zip(*tmp)]
            result.append(r)
    return result


text2 = []
choice = input("Enter SUM or AVG please: ")
text = input("Enter TEXT: ")
text2 = [text]
print(text2)
x = emmbedding(withoutpunc, choice)
y = []
for i in labels:
    y.append([i])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
model = LogisticRegression(class_weight='balanced', C=1e7)
model.fit(X_train, y_train)
yPrediction = model.predict(X_test)
print(yPrediction)
acc =accuracy_score(y_test, yPrediction)
print(acc)
print(acc*100)


def predictt(text, m):
    arroftext=[text]
    print(arroftext)
    dicpredict = emmbedding(arroftext, choice)
    k = []
    #print (len(dicpredict))
    #print (dicpredict)
    for i in dicpredict:
        k.append([i])
    o = m.predict(dicpredict)
    print(o)


predictt(text2, model)

#drink and then drive
#but my biggest kudos go out to sagemiller