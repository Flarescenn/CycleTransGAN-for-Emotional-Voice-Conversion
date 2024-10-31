import torch
import torch.nn as nn
import torch.nn.functional as F


activation_fuc = F.leaky_relu

def gated_linear_layer(inputs, gates):
    return inputs * torch.sigmoid(gates)

def instance_norm_layer(inputs, epsilon=1e-06, activation_fn=None):
    norm_layer = nn.InstanceNorm1d(inputs.shape[1], eps=epsilon)
    outputs = norm_layer(inputs)
    if activation_fn:
        outputs = activation_fn(outputs)
    return outputs

def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same', activation=None):
    conv_layer = nn.Conv1d(inputs.shape[1], filters, kernel_size, stride=strides, padding=padding)
    outputs = conv_layer(inputs)
    if activation:
        outputs = activation(outputs)
    return outputs

def conv2d_layer(inputs, filters, kernel_size, strides, padding='same', activation=None):
    conv_layer = nn.Conv2d(inputs.shape[1], filters, kernel_size, stride=strides, padding=padding)
    outputs = conv_layer(inputs)
    if activation:
        outputs = activation(outputs)
    return outputs

def residual1d_block(inputs, filters=1024, kernel_size=3, strides=1):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = instance_norm_layer(h1, activation_fn=activation_fuc)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = instance_norm_layer(h1_gates, activation_fn=activation_fuc)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    
    h2 = conv1d_layer(h1_glu, filters // 2, kernel_size, strides, activation=activation_fuc)
    h2_norm = instance_norm_layer(h2, activation_fn=F.leaky_relu)
    
    return inputs + h2_norm

def downsample1d_block(inputs, filters, kernel_size, strides):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = instance_norm_layer(h1, activation_fn=activation_fuc)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = instance_norm_layer(h1_gates, activation_fn=activation_fuc)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    
    return h1_glu

def downsample2d_block(inputs, filters, kernel_size, strides):
    h1 = conv2d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = instance_norm_layer(h1, activation_fn=activation_fuc)
    h1_gates = conv2d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = instance_norm_layer(h1_gates, activation_fn=activation_fuc)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    
    return h1_glu

def upsample1d_block(inputs, filters, kernel_size, strides, shuffle_size=2):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_shuffle = pixel_shuffler(h1, shuffle_size)
    h1_norm = instance_norm_layer(h1_shuffle, activation_fn=activation_fuc)

    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_shuffle_gates = pixel_shuffler(h1_gates, shuffle_size)
    h1_norm_gates = instance_norm_layer(h1_shuffle_gates, activation_fn=activation_fuc)

    return gated_linear_layer(h1_norm, h1_norm_gates)

def pixel_shuffler(inputs, shuffle_size=2):
    batch_size, channels, width = inputs.shape
    oc = channels // shuffle_size
    ow = width * shuffle_size
    return inputs.view(batch_size, oc, ow)

class GeneratorGatedCNN(nn.Module):
    def __init__(self):
        super(GeneratorGatedCNN, self).__init__()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)

        h1 = conv1d_layer(inputs, filters=128, kernel_size=15, activation=activation_fuc)
        h1_gates = conv1d_layer(inputs, filters=128, kernel_size=15, activation=activation_fuc)
        h1_glu = gated_linear_layer(h1, h1_gates)

        d1 = downsample1d_block(h1_glu, filters=256, kernel_size=5, strides=2)
        d2 = downsample1d_block(d1, filters=512, kernel_size=5, strides=2)

        r1 = residual1d_block(d2, filters=1024)
        r2 = residual1d_block(r1, filters=1024)
        r3 = residual1d_block(r2, filters=1024)
        r4 = residual1d_block(r3, filters=1024)
        r5 = residual1d_block(r4, filters=1024)
        r6 = residual1d_block(r5, filters=1024)

        u1 = upsample1d_block(r6, filters=1024, kernel_size=5, strides=1, shuffle_size=2)
        u2 = upsample1d_block(u1, filters=512, kernel_size=5, strides=1, shuffle_size=2)

        o1 = conv1d_layer(u2, filters=10, kernel_size=15, activation=activation_fuc)
        o2 = o1.permute(0, 2, 1)

        return o2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)

        h1 = conv1d_layer(inputs, filters=64, kernel_size=3, activation=F.leaky_relu)
        h1_gates = conv1d_layer(inputs, filters=64, kernel_size=3, activation=F.leaky_relu)
        h1_glu = gated_linear_layer(h1, h1_gates)

        d1 = downsample1d_block(h1_glu, filters=128, kernel_size=3, strides=2)
        d2 = downsample1d_block(d1, filters=256, kernel_size=3, strides=2)
        d3 = downsample1d_block(d2, filters=512, kernel_size=6, strides=2)

        o1 = nn.Linear(d3.shape[1] * d3.shape[2], 1)(d3.view(d3.size(0), -1))
        return torch.sigmoid(o1)
