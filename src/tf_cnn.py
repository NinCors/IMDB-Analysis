from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

print(tf.__version__)


def random_data(data,label):
    dic = {}
    for i in range(len(data)):
        dic[i] = data[i]
    
    new_label = []
    data = random.shuffle(data)

    for i in range(len(data)):
        label[i] =dic[data[i]]
    return data,label

def train():
    train_data = np.load("../data/train_data.npy")
    train_labels = np.load("../data/train_data_label.npy")
    test_data = np.load("../data/test_data.npy")
    test_labels = np.load("../data/test_data_label.npy")

    train_labels[train_labels<5] = 0
    train_labels[train_labels>=5] = 1
    test_labels[test_labels<5] = 0
    test_labels[test_labels>=5] = 1



    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=256)

    vocab_size = 200000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    
    model.summary()

 
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    print(train_data.shape)
    print(test_data.shape)
    history = model.fit(train_data,
                    train_labels,
                    epochs=50,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)


train()