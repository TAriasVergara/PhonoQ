# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:34:04 2019

@author: TOMAS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Conv_RNN(nn.Module):
    def __init__(self,batch_size,model_param,model_type=nn.RNN,input_shape=(1,1, 28, 28)):
        """Set the hyper-parameters and build the layers.
        model_type = Default is RNN
        batch_size = batch size  
        model_param = dictionary with the parameters of the model
        """
        super(Conv_RNN, self).__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.sequence_size = model_param['seq_dim']
        self.input_size = model_param['input_dim']
        self.hidden_size = model_param['hidden_dim']
        self.bidirectional = model_param['bidirectional']
        self.output_size = model_param['output_dim']
        self.num_layers = model_param['layer_dim']
        
        #Convolutional layers
        #Convolution only on the features dimension. Not in the sequence!
        nchan_c1 = 8#Number of kernels in conv1
        nchan_c2 = 16#Number of kernels in conv1
        ks = 3
#        self.conv1 = nn.Conv2d(input_shape[1], nchan_c1, kernel_size=(1,ks))
        self.conv1 = nn.Conv2d(input_shape[1], nchan_c1, kernel_size=ks,padding=1)
        self.Bn_c1 = nn.BatchNorm2d(nchan_c1)
        self.MxP_c1 = torch.nn.MaxPool2d((1,2))


#        self.conv2 = nn.Conv2d(nchan_c1, nchan_c2, kernel_size=(1,ks))
        self.conv2 = nn.Conv2d(nchan_c1, nchan_c2, kernel_size=ks,padding=1)
        self.Bn_c2 = nn.BatchNorm2d(nchan_c2)
        self.MxP_c2 = torch.nn.MaxPool2d((1,2))


        self.Relu = torch.nn.LeakyReLU()
        
        #self.LeakyRelu = torch.nn.LeakyReLU()
        #self.seflatt = SelfAttention(nchan_c1,self.LeakyRelu)
        
        #Recompute input size after convolution
        n_size = self._get_conv_output((self.batch_size,input_shape[1],self.sequence_size,self.input_size))
        
        #Number of features
        rnn_INsize = n_size*nchan_c2
        
        #Recurrent network:GRU
        self.model = model_type(rnn_INsize, self.hidden_size,
                            bidirectional=self.bidirectional, 
                            num_layers=self.num_layers,
                            batch_first=True)   
        
        #In case of bidirectional recurrent network
        idx_bi = 1
        if self.bidirectional:
            idx_bi = 2
        #Linear transformation - Affine mapping
        insize = self.hidden_size*idx_bi
        self.Output_layer = nn.Linear(insize, self.output_size)
        self.Bn_l1 = nn.BatchNorm1d(self.sequence_size)
        self.Drop_l1 = nn.Dropout()
        
        #Initialize wieghts
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.Output_layer.weight.data.uniform_(-0.1, 0.1)
        self.Output_layer.bias.data.fill_(0)
     
        # generate input sample and forward to get shape
    def _get_conv_output(self,shape):
#        bs = 1 #batch size
        inputd = Variable(torch.rand(*shape))
        output_feat = self._forward_features(inputd)
        n_size = output_feat.size(3)#output_feat.data.view(inputd.size()[0], -1).size(1)
        return n_size
        
    def _forward_features(self, x):
        """        
        Forward pass (Conv layers)
        """
        #Conv1
        x = self.conv1(x)#Convolution
        x = self.MxP_c1(x)#Max pooling
        x = self.Bn_c1(x)#Batch normalization
        x = self.Relu(x)#Activation function

        #Conv2
        x = self.conv2(x)#Convolution
        x = self.MxP_c2(x)#Max pooling
        x = self.Bn_c2(x)#Batch normalization
        x = self.Relu(x)#Activation function

        #x,_ = self.seflatt(x)
        
        return x
    
    
    def _forward_test(self, x):
        """        
        Forward pass (Conv layers)
        """
        output = {}
        #Conv1
        x = self.conv1(x)#Convolution
        output['conv1'] = x[0].detach().cpu().numpy()
        x = self.MxP_c1(x)#Max pooling
        output['maxP_c1'] = x[0].detach().cpu().numpy()
        x = self.Bn_c1(x)#Batch normalization
        output['BN_c1'] = x[0].detach().cpu().numpy()
        x = self.Relu(x)#Activation function
        output['Relu_c1'] = x[0].detach().cpu().numpy()
#        x,att = self.seflatt(x)
#        output['xSA'] = x[0].detach().cpu().numpy()
#        output['oSA'] = att[0].detach().cpu().numpy()
    
        #Conv1
        x = self.conv2(x)#Convolution
        output['conv2'] = x[0].detach().cpu().numpy()
        x = self.MxP_c2(x)#Max pooling
        output['maxP_c2'] = x[0].detach().cpu().numpy()
        x = self.Bn_c2(x)#Batch normalization
        output['BN_c2'] = x[0].detach().cpu().numpy()
        x = self.Relu(x)#Activation function
        output['Relu_c2'] = x[0].detach().cpu().numpy()
        
        return output
        
    def forward(self, inputs):
        """Define how the data is going to be processed in the network"""
        #Extract features from convolutional layer
        inputs = inputs.view(self.batch_size,self.input_shape[1],self.sequence_size,self.input_size)
        x = self._forward_features(inputs)
        x = x.permute(0,2,1,3)#Permute dimensions to keep one-to-one context
        x = x.contiguous().view(self.batch_size,self.sequence_size,-1)#Concatenate
        
        #Recurrent net
        hiddens, _ = self.model(x)
        #Batch normalization
        outputs = self.Bn_l1(hiddens)
        #Dropout
        outputs = self.Drop_l1(outputs)
        #Affine mapping
        outputs = self.Output_layer(outputs)
        return outputs
 
        
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out * x
        return out,attention