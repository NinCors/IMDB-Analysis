import numpy as np
import sys

class ConvLayer(object):
    def __init__(self, kernal_num,layer_num,  kernal_size, stride = 1, lr = 0.03, pd = 1, reg = 0.75):
        self.name = "Convolutional_layer"
        #initial the kernals with random values 
        self.conv_kernals = np.random.randn(kernal_num,layer_num,kernal_size,kernal_size)
        self.pre_kernals = np.zeros_like(self.conv_kernals)
        #inital bias with random values
        self.bias = np.random.randn(layer_num)
        self.pre_bias = np.zeros_like(self.bias)
        #paramente values
        self.lr = lr
        self.stride = stride
        self.reg = reg
        self.pd = pd
        self.input = None
        
    def forward(self,input):
        self.input = input
        node_num, layer_num, h, w = input.shape
        kernal_num_k, layer_num_k, h_k, w_k = self.conv_kernals.shape
        h_res = int ((2*self.pd + h - h_k) / self.stride) + 1
        w_res = int ((2*self.pd + w - w_k) / self.stride) + 1
        res = np.zeros((node_num,kernal_num_k,h_res,w_res))
        #padding the input value with 0
        input = np.pad(input,((0,), (0,), (self.pd,), (self.pd,)), mode='constant')
        #Scan the input vector
        for i in range(h):
            for j in range(w):
                scanned_area = input[:,:,i*self.stride:i*self.stride+h_k,j*self.stride:j*self.stride+w_k]
                for k in range(kernal_num_k):
                    res[:,k,i,j] = np.sum(scanned_area*self.conv_kernals[k,:,:,:], axis=(1,2,3))
        return res

    def backward():
        return -1

class PoolingLayer:
    def __init__(self, kernal_size, type="max", stride=1):
        self.kernal_size = kernal_size
        self.type = type
        self.name = "Pooling layer"
        self.stride = stride
        self.input = None
    
    def forward(self,input):
        self.input = input
        f,s,h,w = input.shape
        #build the out vector with the right format
        res_h = int((h-self.kernal_size)/self.stride) + 1
        res_w = int((w-self.kernal_size)/self.stride) + 1
        res = np.zeros((f,s,res_h,res_w))
        #scane the input
        for i in range(res_h):
            for j in range(res_w):
                scanned_area = input[:,:,i*self.stride:i*self.stride + self.kernal_size, j*self.stride:j*self.stride+self.kernal_size]
                if(self.type == "max"):
                    res[:,:,i,j] = np.max(scanned_area,axis=(2,3))
        return res

    def backward():
        return -1

class NormalizationLayer:
    def __init__(self, method):
        self.method = method
        self.name = "Normalization layer"
        self.input = None
    
    def forward(self,input):
        if(self.method == 'relu'):
            input[input<0] = 0
            self.input = input
        return input

    def backward():
        return -1

class FlattenLayer:
    def __init__(self):
        self.name = "Flatten layer"
        self.input = None
        self.f = 0
        self.s = 0
        self.h = 0
        self.w = 0
    
    def forward(self, input):
        self.f,self.s,self.h, self.w = input.shape
        return input.reshape(self.f, self.s * self.h * self.w)

    def backward(self,input):
        return -1

class FulllyConnectedLayer:
    def __init__(self, w_h, w_w, lr = 0.01, reg=0.75, std=1e-4):
        self.weights = np.random.randn(w_h,w_w) * std
        self.bias = np._zeros(w_w)
        self.lr = lr
        self.reg= reg
        self.pre_w = np.zeros_like(self.weights)
        self.pre_b = np.zeros_like(self.bias)
        self.input = None

    def forward(self, input):
        self.input = input
        return input.dot(self.weights)+self.bias

    def backward():
        return -1


    