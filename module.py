import torch
import torch.nn as nn
import torch.nn.functional as F

activation_fuc = F.leaky_relu
# activation_fuc = F.elu

def gated_linear_layer(inputs, gates):
    return inputs * torch.sigmoid(gates)

class InstanceNormLayer(nn.Module):
    def __init__(self, inputs, epsilon=1e-06, activation_fn=None):
        super(InstanceNormLayer, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(inputs.shape[1], eps=epsilon, affine=False)
        self.activation_fn = activation_fn

    def forward(self, inputs):
        x = self.instance_norm(inputs)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same', activation=None, kernel_initializer=None):
    conv_layer = nn.Conv1d(inputs.shape[1], filters, kernel_size, stride=strides, padding=padding)
    x = conv_layer(inputs)
    if activation:
        x = activation(x)
    return x

def conv2d_layer(inputs, filters, kernel_size, strides, padding='same', activation=None, kernel_initializer=None):
    conv_layer = nn.Conv2d(inputs.shape[1], filters, kernel_size, stride=strides, padding=padding)
    x = conv_layer(inputs)
    if activation:
        x = activation(x)
    return x

def residual1d_block(inputs, filters=1024, kernel_size=3, strides=1):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = InstanceNormLayer(activation_fn=activation_fuc)(h1)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = InstanceNormLayer(activation_fn=activation_fuc)(h1_gates)
    h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)
    h2 = conv1d_layer(h1_glu, filters // 2, kernel_size, strides, activation=activation_fuc)
    h2_norm = InstanceNormLayer(activation_fn=F.leaky_relu)(h2)
    return inputs + h2_norm

def downsample1d_block(inputs, filters, kernel_size, strides):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = InstanceNormLayer(activation_fn=activation_fuc)(h1)
    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = InstanceNormLayer(activation_fn=activation_fuc)(h1_gates)
    return gated_linear_layer(h1_norm, h1_norm_gates)

def downsample2d_block(inputs, filters, kernel_size, strides):
    h1 = conv2d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm = InstanceNormLayer(activation_fn=activation_fuc)(h1)
    h1_gates = conv2d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_norm_gates = InstanceNormLayer(activation_fn=activation_fuc)(h1_gates)
    return gated_linear_layer(h1_norm, h1_norm_gates)

def upsample1d_block(inputs, filters, kernel_size, strides, shuffle_size=2):
    h1 = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_shuffle = pixel_shuffler(h1, shuffle_size)
    h1_norm = InstanceNormLayer(activation_fn=activation_fuc)(h1_shuffle)

    h1_gates = conv1d_layer(inputs, filters, kernel_size, strides, activation=activation_fuc)
    h1_shuffle_gates = pixel_shuffler(h1_gates, shuffle_size)
    h1_norm_gates = InstanceNormLayer(activation_fn=activation_fuc)(h1_shuffle_gates)

    return gated_linear_layer(h1_norm, h1_norm_gates)

def pixel_shuffler(inputs, shuffle_size=2):
    batch_size, channels, width = inputs.size()
    out_channels = channels // shuffle_size
    out_width = width * shuffle_size
    return inputs.view(batch_size, out_channels, out_width)

class GeneratorGatedCNN(nn.Module):
    def __init__(self):
        super(GeneratorGatedCNN, self).__init__()
        # Define layers here
        self.conv1 = nn.Conv1d(128, 128, 15, 1)
        self.downsample1 = downsample1d_block
        self.residual1 = residual1d_block
        self.upsample1 = upsample1d_block
        self.conv_output = nn.Conv1d(512, 24, 15, 1)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        h1 = self.conv1(inputs)
        h1_glu = gated_linear_layer(h1, h1)
        d1 = self.downsample1(h1_glu, 256, 5, 2)
        r1 = self.residual1(d1, 1024, 3, 1)
        # Transformer layer would go here
        u1 = self.upsample1(r1, 512, 5, 1, 2)
        o1 = self.conv_output(u1)
        return o1.transpose(1, 2)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 2))
        self.dense = nn.Linear(1024, 1)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        h1 = self.conv1(inputs)
        h1_glu = gated_linear_layer(h1, h1)
        # Downsample steps go here
        d1 = downsample2d_block(h1_glu, 256, (3, 3), (2, 2))
        o1 = torch.sigmoid(self.dense(d1))
        return o1
