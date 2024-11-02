import torch
import torch.nn as nn
import torch.nn.functional as F
from module import activation_fuc, conv1d_layer, conv2d_layer, gated_linear_layer, downsample1d_block, downsample2d_block, residual1d_block, upsample1d_block

class generator_gatedcnn(nn.Module):
    def __init__(self, input_channels):
        super(generator_gatedcnn, self).__init__()
        
       
       # self.input_transpose = lambda x: x.permute(0, 2, 1)
        
        # Initial conv layer and gated conv layer
        self.h1_conv = conv1d_layer(inputs=input_channels, filters=128, kernel_size=15, strides=1, activation=True)
        self.h1_conv_gates = conv1d_layer(inputs=input_channels, filters=128, kernel_size=15, strides=1, activation=True)
        
        # Downsampling layers
        self.downsample1 = downsample1d_block(128, filters=256, kernel_size=5, strides=2)
        self.downsample2 = downsample1d_block(256, filters=512, kernel_size=5, strides=2)
        #self.downsample3 = downsample1d_block(512, filters=1024, kernel_size=5, strides=2)
        
        # Residual blocks
        self.residual1 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        self.residual2 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        self.residual3 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        self.residual4 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        self.residual5 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        self.residual6 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)
        
        # Upsampling layers
        self.upsample1 = upsample1d_block(512, filters=1024, kernel_size=5, strides=1, shuffle_size=2)
        self.upsample2 = upsample1d_block(512, filters=512, kernel_size=5, strides=1, shuffle_size=2)
        
        # Final convolution and output transpose
        self.output_conv = conv1d_layer(256, filters=10, kernel_size=15, strides=1, activation=True)
        #self.output_transpose = lambda x: x.permute(0, 2, 1)
        
    def forward(self, inputs):
        #x = self.input_transpose(inputs)  # Transpose inputs
        
        
        h1 = self.h1_conv(inputs)
        h1_gates = self.h1_conv_gates(inputs)
        h1_glu = gated_linear_layer(h1, h1_gates)  
        #print(h1_glu.shape)
        # Downsampling layers
        d1 = self.downsample1(h1_glu)
      
        d2 = self.downsample2(d1)
        #d3 = self.downsample3(d2)
       
        # Residual blocks
        r1 = self.residual1(d2)
        r2 = self.residual2(r1)
        r3 = self.residual3(r2)
        r4 = self.residual4(r3)
        r5 = self.residual5(r4)
        r6 = self.residual6(r5)
        #print(r6.shape)
        # Upsampling layers
        u1 = self.upsample1(r6)
        u2 = self.upsample2(u1)
        #print(u2.shape)
        
        o1 = self.output_conv(u2)
        #o2 = self.output_transpose(o1)  # Final transpose to match output dimensions
        #print(o1.shape)
        return o1
    

class _discriminator(nn.Module):
    def __init__(self, inputs):
        super(_discriminator, self).__init__()
        
        
        self.expand_dims = lambda x: x.unsqueeze(2)
        
        
        self.h1_conv = conv2d_layer(inputs, filters=16, kernel_size=(3, 3), strides=(1, 2), activation=True)
        self.h1_conv_gates = conv2d_layer(inputs, filters=16, kernel_size=(3, 3), strides=(1, 2), activation=True)
        
        # Downsampling blocks
        self.downsample1 = downsample2d_block(16, filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.downsample2 = downsample2d_block(32, filters=32, kernel_size=(3, 3), strides=(2, 2))
        self.downsample3 = downsample2d_block(32, filters=64, kernel_size=(6, 3), strides=(1, 2))
        
        # Final dense layer
        self.output_dense = nn.Linear(64, 1)  

    def forward(self, inputs):
        x = self.expand_dims(inputs)  

        
        h1 = self.h1_conv(x)
        h1_gates = self.h1_conv_gates(x)
        h1_glu = gated_linear_layer(h1, h1_gates)  
        
        # Downsampling layers
        d1 = self.downsample1(h1_glu)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        
        # Flatten and apply final dense layer
        d3_flattened = d3.view(d3.size(0), -1)  # Flatten for dense layer
        o1 = torch.sigmoid(self.output_dense(d3_flattened))
        
        return o1

class discriminator(nn.Module):
    def __init__(self, inputs):
        super(discriminator, self).__init__()
        
        # Transpose to match TensorFlow input shape
        #self.input_transpose = lambda x: x.permute(0, 2, 1)
     
        self.h1_conv = conv1d_layer(inputs, filters=64, kernel_size=3, strides=1, activation=True)
        self.h1_conv_gates = conv1d_layer(inputs, filters=64, kernel_size=3, strides=1, activation=True)
        
        # Downsampling layers
        self.downsample1 = downsample1d_block(64, filters=128, kernel_size=3, strides=2)
        self.downsample2 = downsample1d_block(128, filters=256, kernel_size=3, strides=2)
        self.downsample3 = downsample1d_block(256, filters=512, kernel_size=6, strides=4)
        self.downsample4 = downsample1d_block(512, filters=512, kernel_size=6, strides=4)
        # Final dense layer
        self.output_dense = nn.Linear(1024, 1)  

    def forward(self, inputs):
        #x = self.input_transpose(inputs)  
        x=inputs
       
        h1 = self.h1_conv(x)
        h1_gates = self.h1_conv_gates(x)
        h1_glu = gated_linear_layer(h1, h1_gates)  
        
        # Downsampling layers
        d1 = self.downsample1(h1_glu)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        #d3=d2
        # Flatten and apply final dense layer
        d4_flattened = d4.view(d4.size(0), -1)  # Flatten for dense layer
        o1 = torch.sigmoid(self.output_dense(d4_flattened))
        
        return o1
