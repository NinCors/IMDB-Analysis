import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

words_dict = "../imdb_data/imdb.vocab"
train_data_dir_neg = "../imdb_data/train/neg"
train_data_dir_pos = "../imdb_data/train/pos"

# Handling data
#   Load the dictionary from words_dict
#   Scan all the file_name from the review directory
#       Extract the review score from file name, and save it in the array
#       Read the file and convert the content to number based on the dictionary we got
#           Padding the data to 256
#           Embedding the data


#Using the index from tensorflow
#Credit to https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_text_classification.ipynb?pli=1#scrollTo=tr5s_1alpzop
dictionary = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def get_list(data_dirctory):
    dirct = os.listdir(data_dirctory)
    return dirct

def embedding_words(words):

def padding(words, size=256):
    
def get_review(file_name):

def get_data(data_dirctory):
    train_data = []
    train_label = []
    lists = get_list(data_dirctory)
    for filename in lists:
        #extract file name from
        tmp = filename.split('-')
        tmp = tmp[1].split('.')
        train_label.append(int(tmp[0]))
        review_vector = get_review(filename)
        train_data.append(review_vector)
    print(train_data.shape)
    return train_data, train_label






