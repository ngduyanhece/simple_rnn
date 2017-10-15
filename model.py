import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import string
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,GRU,SimpleRNN

def remove_punctuation(s):
    return s.translate(string.punctuation)

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('./robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx

def build_model(max_features):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(SimpleRNN(1,activation='relu'))
    model.add(Dense(max_features, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

sentences, word2idx = get_robert_frost()
rnn_model = build_model(len(word2idx))
for i in range(2000):
    X = shuffle(sentences)
    for j in range(len(X)):
        input_sequence = [0] + X[j]
        output_sequence = X[j] + [1]
        rnn_model.fit(input_sequence,output_sequence,epochs=1,verbose=0)
        if j == (len(X) - 1):
            loss, acc = rnn_model.evaluate(input_sequence, output_sequence, verbose=0)
            print('epoch: {} --- loss: {} --- acc: {}'.format(i,loss, acc))
sample = [0] + sentences[0]
print('sample input: {}'.format(sample))
labels = rnn_model.predict_classes(sample)
print('predict sentences: {}'.format(labels))