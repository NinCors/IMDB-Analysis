import numpy as np
import sys
import random


#CNN:
#   ConvLayer -> Get the result by multiply with the feature table
#   PoolingLayer -> Reduce the map and keep the important information
#   NormalizationLayer -> Activation function to convert the data between 0-1 to increase the speed
#   FulllyConnectedLayer -> Every value in the result get a vote


class CNN:
    def __init__(self):
        self.NAME = "CNN"
        self.CNN_LAYERS = []
    
    # Appedn a new layer into the current CNN layers
    def append_layer(self, layer):
        self.CNN_LAYERS.append(layer)

    # Trainning process contains the forward and backward function
    def train(trainData,trainLabel,batch_size,iteration):
        return -1

    # Perform the forward operation for all the layers
    def forward(self,data,label):
        return -1

    # Perform the backward operation for all the layers
    def backward(self,label):
        return -1

    # Use the current layers to predict the value
    def predict(self,data,label):
        return -1

