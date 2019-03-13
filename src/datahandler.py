import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import random

print(tf.__version__)

words_dict = "../imdb_data/imdb.vocab"
train_data_dir_neg = "../imdb_data/train/neg/"
train_data_dir_pos = "../imdb_data/train/pos/"
test_data_dir_neg = "../imdb_data/test/neg/"
test_data_dir_pos = "../imdb_data/test/pos/"

# Handling data
#   Load the dictionary from words_dict
#   Scan all the file_name from the review directory
#       Extract the review score from file name, and save it in the array
#       Read the file and convert the content to number based on the dictionary we got
#           Padding the data to 256
#           Embedding the data


#Using the index from tensorflow
#Credit to https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb?pli=1#scrollTo=tr5s_1alpzop
word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def encode_review(text):
    for i in range(len(text)):
        if(text[i].lower() in word_index):
            text[i] = word_index[text[i].lower()]
        else:
            text[i] = 0
    return text

def decode_review(text):
    for i in range(len(text)):
        if(text[i] in reverse_word_index):
            text[i] = reverse_word_index[text[i]]
        else:
            text[i] = "UNK!!"
    return text

def get_list(data_dirctory):
    dirct = os.listdir(data_dirctory)
    return dirct

def embedding_words(words):
    #Credit to https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb?pli=1#scrollTo=tr5s_1alpzop
    vocab_size = 100000
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.compile('rmsprop', 'mse')
    res = model.predict(words)
    return res

def padding(words, size=256):
    size = len(words)
    while(size < 256):
        words.append(0)
        size += 1
    return words

def get_review(file_name):
    with open(file_name) as f:
        content = f.readlines()
    #print(content[0])
    review = content[0].split()

    for i in range(len(review)):
        pos = len(review[i])
        for c in range(len(review[i])):
            if(not review[i][c].isalpha()):
                pos = c
                break
        review[i] = review[i][0:pos]

    #sprint(review)
    review = encode_review(review)
    #print(np.shape(review))
    #review = padding(review)
    #print(np.shape(review))
    #print (review)
    return review


def get_data(data_dirctory):
    train_data = []
    train_label = []
    lists = get_list(data_dirctory)
    for filename in lists:
        #extract file name from
        print("OPen file {}".format(filename))
        tmp = filename.split('_')
        tmp = tmp[1].split('.')
        train_label.append(int(tmp[0]))
        review_vector = get_review(data_dirctory+filename)
        train_data.append(review_vector)
        
    #print(train_data.shape)
    return train_data, train_label

def train_dataset(file_name):
    start_time = time.time()

    pos_train_data, pos_train_label = get_data(train_data_dir_pos)
    neg_data,neg_label = get_data(train_data_dir_neg)
    data = pos_train_data + neg_data
    label = pos_train_label + neg_label

    print(np.shape(data))
    print(np.shape(label))
    np.save("../data/"+file_name,data)
    np.save("../data/"+file_name+"_label",label)
    
    #data = embedding_words(data)
    #np.save("../data/"+file_name+"_embed",data)

    elapsed_time = time.time() - start_time
    print("It took {} to finish this shit".format(elapsed_time))
    print("Saved the data into files")


def test_dataset(file_name):
    start_time = time.time()

    pos_train_data, pos_train_label = get_data(test_data_dir_pos)
    neg_data,neg_label = get_data(test_data_dir_neg)
    data = pos_train_data + neg_data
    label = pos_train_label + neg_label

    print(np.shape(data))
    print(np.shape(label))
    np.save("../data/"+file_name,data)
    np.save("../data/"+file_name+"_label",label)
    
    #data = embedding_words(data)
    #np.save("../data/"+file_name+"_embed",data)

    elapsed_time = time.time() - start_time
    print("It took {} to finish this shit".format(elapsed_time))
    print("Saved the data into files")



train_dataset("train_data")
train_dataset("test_data")

#test_data = np.load("neg_train_data_embed.npy")
#print(test_data.shape)
#test_data = np.load("neg_train_data_embed.npy")
#print(test_data.shape)