import torch
import torch.nn as nn
import torch.nn.functional as F
from module import activation_fuc, conv1d_layer, conv2d_layer, gated_linear_layer, downsample1d_block, downsample2d_block, residual1d_block, upsample1d_block

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
#class _discriminator
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
