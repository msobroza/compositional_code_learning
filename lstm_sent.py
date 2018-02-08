from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras import optimizers
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm



# Get imdb mapping
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

def gen_dictionary(data_train, data_test, embedding_index):
    global index_word
    new_word_index = dict()
    count = 0
    new_data_train = list()
    for seq in data_train:
        new_seq = list()
        for w_i in seq:
            w = index_word[w_i]
            if w in embedding_index:
                if w not in new_word_index:
                    count+=1
                    new_word_index[w]=count
                new_seq.append(new_word_index[w])
        new_data_train.append(new_seq)
    new_data_test=list()
    for seq in data_test:
        new_seq = list()
        for w_i in seq:
            w = index_word[w_i]
            if w in new_word_index:
               new_seq.append(new_word_index[w])
        new_data_test.append(new_seq)
    return new_data_train, new_data_test, new_word_index


# Get GloVe mapping

GLOVE_DIR = './'
#FILE_PATH = 'glove.42B.300d.txt'
FILE_PATH = 'dense_70K_64x8d_2.txt'
embedding_word= dict() 
f = open(os.path.join(GLOVE_DIR, FILE_PATH),'r')
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs= np.array(values[1:], dtype='float32')
    embedding_word[word] = coefs
    if count % 10000 == 0:
       print(count)
    count+=1
f.close()
print('GloVe vectors loaded...')

maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=maxlen)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
x_train, x_test, word_index = gen_dictionary(x_train, x_test, embedding_word)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
index_word = {v: k for k, v in word_index.items()}
max_features = len(word_index)
print('number of max_features', max_features)
embeddings_matrix = np.zeros((max_features+1, 300), dtype=np.float32)

for i in index_word.keys():
    embeddings_matrix[i,:]=embedding_word[index_word[i]]


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('embedding_matrix:', embeddings_matrix.shape)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features+1, 300, embeddings_initializer=tf.constant_initializer(embeddings_matrix), trainable=False))
model.add(LSTM(150))
model.add(Dense(1, activation='sigmoid'))


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))

acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print(acc)
print(type(acc))
