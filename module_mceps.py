import torch
import torch.nn as nn
import torch.nn.functional as F

# activation_func = F.relu
activation_func = F.leaky_relu

def gated_linear_layer(inputs, gates, name=None):
    return inputs * torch.sigmoid(gates)

class InstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(InstanceNorm1d, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.instance_norm(x)

def conv1d_layer(inputs, out_channels, kernel_size, stride=1, padding='same', activation=None):
    if padding == 'same':
        padding = kernel_size // 2
    conv = nn.Conv1d(inputs.size(1), out_channels, kernel_size, stride, padding)
    x = conv(inputs)
    if activation:
        x = activation(x)
    return x

def conv2d_layer(inputs, out_channels, kernel_size, stride, padding='same', activation=None):
    if padding == 'same':
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    conv = nn.Conv2d(inputs.size(1), out_channels, kernel_size, stride, padding)
    x = conv(inputs)
    if activation:
        x = activation(x)
    return x

def residual1d_block(inputs, filters=1024, kernel_size=3, stride=1, name_prefix='residual_block_'):
    h1 = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm = InstanceNorm1d(filters)(h1)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm_gates = InstanceNorm1d(filters)(h1_gates)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    h2 = conv1d_layer(h1_glu, filters // 2, kernel_size, stride, activation=activation_func)
    h2_norm = InstanceNorm1d(filters // 2)(h2)
    return inputs + h2_norm

def downsample1d_block(inputs, filters, kernel_size, stride, name_prefix='downsample1d_block_'):
    h1 = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm = InstanceNorm1d(filters)(h1)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm_gates = InstanceNorm1d(filters)(h1_gates)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    return h1_glu

def downsample2d_block(inputs, filters, kernel_size, stride, name_prefix='downsample2d_block_'):
    h1 = conv2d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm = InstanceNorm1d(filters)(h1.view(h1.size(0), h1.size(1), -1)).view_as(h1)
    h1_gates = conv2d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_norm_gates = InstanceNorm1d(filters)(h1_gates.view(h1_gates.size(0), h1_gates.size(1), -1)).view_as(h1_gates)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    return h1_glu

def upsample1d_block(inputs, filters, kernel_size, stride, shuffle_size=2, name_prefix='upsample1d_block_'):
    h1 = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_shuffle = pixel_shuffler(h1, shuffle_size)
    h1_norm = InstanceNorm1d(filters // shuffle_size)(h1_shuffle)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, stride, activation=activation_func)
    h1_shuffle_gates = pixel_shuffler(h1_gates, shuffle_size)
    h1_norm_gates = InstanceNorm1d(filters // shuffle_size)(h1_shuffle_gates)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    return h1_glu

def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = inputs.size(0)
    c = inputs.size(1)
    w = inputs.size(2)
    oc = c // shuffle_size
    ow = w * shuffle_size
    outputs = inputs.view(n, oc, ow)
    return outputs

class GeneratorGatedCNN(nn.Module):
    def __init__(self):
        super(GeneratorGatedCNN, self).__init__()
        self.conv1 = nn.Conv1d(24, 128, kernel_size=15, stride=1, padding=7)
        self.conv1_gates = nn.Conv1d(24, 128, kernel_size=15, stride=1, padding=7)

    def forward(self, inputs):
        # inputs has shape [batch_size, num_features, time]
        # we need to convert it to [batch_size, time, num_features] for 1D convolution
        inputs = inputs.transpose(1, 2)

        h1 = self.conv1(inputs)
        h1_gates = self.conv1_gates(inputs)
        h1_glu = gated_linear_layer(h1, h1_gates)

        # Downsample
        d1 = downsample1d_block(h1_glu, 256, 5, 2)
        d2 = downsample1d_block(d1, 512, 5, 2)

        # Residual blocks
        r1 = residual1d_block(d2, 1024, 3, 1)
        r2 = residual1d_block(r1, 1024, 3, 1)

        # Upsample
        u1 = upsample1d_block(r2, 1024, 5, 1, 2)
        u2 = upsample1d_block(u1, 512, 5, 1, 2)

        o1 = conv1d_layer(u2, 24, 15, 1, activation=activation_func)
        o2 = o1.transpose(1, 2)

        return o2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.conv1_gates = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
        self.dense = nn.Linear(1024, 1)

    def forward(self, inputs):
        # inputs has shape [batch_size, num_features, time]
        # we need to add channel for 2D convolution [batch_size, 1, num_features, time]
        inputs = inputs.unsqueeze(1)

        h1 = self.conv1(inputs)
        h1_gates = self.conv1_gates(inputs)
        h1_glu = gated_linear_layer(h1, h1_gates)

        # Downsample
        d1 = downsample2d_block(h1_glu, 256, (3, 3), (2, 2))
        d2 = downsample2d_block(d1, 512, (3, 3), (2, 2))
        d3 = downsample2d_block(d2, 1024, (6, 3), (1, 2))

        # Output
        o1 = self.dense(d3.view(d3.size(0), -1))
        return torch.sigmoid(o1)