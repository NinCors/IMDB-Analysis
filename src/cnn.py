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
    def append(self, layer):
        self.CNN_LAYERS.append(layer)

    # Trainning process contains the forward and backward function
    def train(trainData,trainLabel, testData, testLabel, batch_size=512,epoch=100):
        data_size = trainData.shape[0]
        for e in range(epoch):
            print(str(time.clock()) + ': Epoch = ' + str(e))
            for i in range(0, data_size, batch_size):
                bound = i+batch_size
                if(bound > data_size):
                    bound = data_size
                tra_data = trainData[i:i+batch_size]
                tra_label = trainLabel[i:i+batch_size]
                loss = self.forward(tra_data,tra_label)
                print("Loss: {}".format(loss))
                self.backward(tra_label)
            print("Training data accuracy : {}".format(self.predict(trainData,trainLabel)))
            print("Testing data accuracy : {}".format(self.predict(testData,testLabel)))
        print("Finish")

    # Perform the forward operation for all the layers
    def forward(self,data,label):
        layer_size = len(self.CNN_LAYERS)
        input_data = data
        for i in range(layer_size):
            output_data = self.CNN_LAYERS[i].forward(input_data)
            input_data = output_data
        # Loss function
        loss = -np.sum(np.log(output_data[range(len(output_data.shape[0])), list(label)]))/outout_data.shape[0]
        return loss

    # Perform the backward operation for all the layers
    def backward(self,label):
        input_residual = label
        for i in range(len(self.CNN_LAYERS)):
            output_residual = self.CNN_LAYERS.backward(input_residual)
            input_residual = output_residual

    # Use the current layers to predict the value
    def predict(self,data,label):
        layer_size = len(self.CNN_LAYERS)
        input_data = data
        for i in range(layer_size):
            output_data = self.CNN_LAYERS[i].forward(input_data)
            input_data = output_data
        out_label = np.argmax(output_data, axis=1)
        return float(np.sum(out_label == label))/ float(out_label.shape[0])

