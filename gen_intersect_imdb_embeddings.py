from keras.datasets import imdb
import os
import numpy as np
from tqdm import tqdm

# Get imdb mapping
word_index = imdb.get_word_index()
index_word = {v: k for k, v in word_index.items()}

count = 0
for w in word_index.keys():
    count+=1


# Get GloVe mapping

GLOVE_DIR = './'
DESTINATION_PATH= './'
embedding_index= dict() 
f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'),'r')
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs= np.array(values[1:], dtype='float32')
    embedding_index[word] = coefs
    if count % 10000 == 0:
       print(count)
    count+=1
f.close()
print('GloVe vectors loaded...')


# Save file containing corpus intersection 
f = open(os.path.join(DESTINATION_PATH,'glove.42B.300d.txt'), 'w')

count = 0
for w in tqdm(word_index.keys()):
    if w in embedding_index:
        vec_str= " ".join(map(str, embedding_index[w]))
        f.write(w+' '+vec_str+'\n')
        count+=1

f.close()
print(count)
