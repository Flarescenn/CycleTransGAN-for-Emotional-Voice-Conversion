import torch
import torch.nn as nn
import torch.nn.functional as F

activation_fuc = F.leaky_relu

def gated_linear_layer(inputs, gates):
    """
  Gated Linear Unit (GLU) layer.
  Element-wise multiplication of inputs and gates, and a sigmoid gate to control the flow of information.

  Args:
    inputs: Input tensor.
    gates: Gate tensor.

  Returns:
    Output tensor after applying the GLU operation.
  """
    return inputs * torch.sigmoid(gates)


class instance_norm_layer(nn.Module):
    def __init__(self, inputs, epsilon=1e-06, activation=None, name = None):
        super(instance_norm_layer, self).__init__()
        self.instance_norm = nn.InstanceNorm1d(inputs, eps=epsilon, affine=False)
        self.activation = activation

    def forward(self, inputs):
        x = self.instance_norm(inputs)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x
    

class conv1d_layer(nn.Module):
    def __init__(
        self, 
        inputs, 
        filters, 
        kernel_size, 
        strides=1, 
        padding='same', 
        activation=None,
        kernel_initializer=None,
        name=None):

        super(conv1d_layer, self).__init__()

        if padding == 'same':
            padding_size = (kernel_size - 1) // 2
        else:  # 'valid'
            padding_size = 0

        self.conv = nn.Conv1d(
        in_channels=inputs,
        out_channels=filters,
        kernel_size=kernel_size,
        stride=strides,
        padding=padding_size
        )

        self.activation = activation

        def forward(self, x):
            out = self.conv(x)
            if self.activation:
                out = activation_fuc(out)
            return out
        
class conv2d_layer(nn.Module):
    def __init__(
        self, 
        inputs, 
        filters, 
        kernel_size, 
        strides, 
        padding='same', 
        activation=None,
        kernel_initializer=None,
        name=None):

        super(conv2d_layer, self).__init__()

        if padding == 'same':
            padding_height = (kernel_size[0] - 1) // 2
            padding_width = (kernel_size[1] - 1) // 2
            padding_size= (padding_height, padding_width)
        else:  
            padding_size = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)

        self.conv = nn.Conv2d(
        in_channels=inputs,
        out_channels=filters,
        kernel_size=kernel_size,
        stride=strides,
        padding=padding_size
        )

        self.activation = activation

        def forward(self, x):
            out = self.conv(x)
            if self.activation:
                out = activation_fuc(out)
            return out
        
            
class pixel_shuffler(nn.Module):
    """
    Pixel Shuffling (Sub-Pixel Convolution)
    Tensor of shape (N, W, C) ---> Tensor of Shape (N, W*r, C/r)
    For upsampling.

    Args: 
        shuffle_size (int): The factor by which to increase the temporal dimension
    Shape:
        - Input: (N, W, C) where
            * N is the batch size
            * W is the width
            * C is the number of channels (must be divisible by shuffle_size)
        - Output: (N, W*shuffle_size, C/shuffle_size)
    """
    def __init__(self, shuffle_size=2):
        super(pixel_shuffler, self).__init__()
        self.shuffle_size = shuffle_size
    
    def forward(self, x):
        N = x.size(0)
        W = x.size(1)
        C = x.size(2)

        out_channels = C// self.shuffle_size
        out_width = W * self.shuffle_size

        return x.reshape(N, out_width, out_channels)


class residual1d_block(nn.Module):
    """
    Architecture:
    Input → (Conv1D → InstanceNorm → GLU) → Conv1D → InstanceNorm + Input
    """
    def __init__(self, in_channels, filters=1024, kernel_size=3, strides=1, name_prefix='residual_block_'):
        super(residual1d_block, self).__init__()

        self.conv1 = conv1d_layer(
            inputs=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=True,  # Will use the leaky_relu
            name = name_prefix + 'h1conv'
        )
        self.norm1 = instance_norm_layer(
            inputs=filters,
            activation=True,
            name=name_prefix + 'h1_norm'
        )
        #Parallel path for gates
        self.conv1_gates = conv1d_layer(
            inputs=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=True,
            name=name_prefix + 'h1_gates'
        )
        self.norm1_gates = instance_norm_layer(
            inputs=filters,
            activation=True,
            name=name_prefix + 'h1_norm_gates'
        )

        #Second Conv layer with half the filters
        self.conv2 = conv1d_layer(
            inputs=filters,
            filters=filters // 2,
            kernel_size=kernel_size,
            strides=strides,
            activation=True,
            name=name_prefix + 'h2_conv'
        )
        self.norm2 = instance_norm_layer(
            inputs=filters // 2,
            activation=True,
            name=name_prefix + 'h2_norm'
        )
        
    
    def forward(self, x):
        # First path
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)

        # Gate path
        h1_gates = self.conv1_gates(x)
        h1_norm_gates = self.norm1_gates(h1_gates)

        # Apply existing GLU function
        h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)

        # Second path
        h2 = self.conv2(h1_glu)
        h2_norm = self.norm2(h2)

        # Add skip connection
        h3 = x + h2_norm
        
        return h3


class downsample1d_block(nn.Module):
    '''
    Architecture:
        Input → (Conv1D → InstanceNorm) ⨁ (Conv1D → InstanceNorm) → GLU
    '''
    def __init__(self, in_channels, filters, kernel_size, strides):
        super(downsample1d_block, self).__init__()
        self.conv1 = conv1d_layer(
            inputs=in_channels, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            activation=True  
        )
        self.norm1 = instance_norm_layer(
            inputs=filters,
            activation=True,
        )
        self.conv1_gates = conv1d_layer(
            inputs=in_channels, 
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            activation=True  # for LeakyReLU
        )
        self.norm1_gates = instance_norm_layer(
            inputs=filters, 
            activation=True
        )

    def forward(self, x):
        # Main Path
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)
        
        # Gating path
        h1_gates = self.conv1_gates(x)
        h1_norm_gates = self.norm1_gates(h1_gates)
        
        # Apply GLU
        out = gated_linear_layer(h1_norm, h1_norm_gates)
        
        return out

class upsample1d_block(nn.Module):
    """
    Architecture:
        Input → (Conv1D → PixelShuffle → InstanceNorm) ⨁ (Conv1D → PixelShuffle → InstanceNorm) → GLU
    """
    def __init__(self, in_channels, filters, kernel_size, strides, shuffle_size=2):
        super(upsample1d_block, self).__init__()
        
        #main path
        self.conv1 = conv1d_layer(
            inputs=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=True  
        )
        self.pixel_shuffle = pixel_shuffler(shuffle_size)
        
        # Instance Normalization for the main path
        self.norm1 = instance_norm_layer(
            inputs=filters // shuffle_size,
            activation=True
        )
        
        # Gating path
        self.conv1_gates = conv1d_layer(
            inputs=in_channels,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=True
        )
        self.pixel_shuffle_gates = pixel_shuffler(shuffle_size)
        self.norm1_gates = instance_norm_layer(
            inputs=filters // shuffle_size,
            activation=True
        )

    def forward(self, x):
        # Main Path
        h1 = self.conv1(x)
        h1_shuffle = self.pixel_shuffle(h1)
        h1_norm = self.norm1(h1_shuffle)
        
        # Gating Path
        h1_gates = self.conv1_gates(x)
        h1_shuffle_gates = self.pixel_shuffle_gates(h1_gates)
        h1_norm_gates = self.norm1_gates(h1_shuffle_gates)
        
        out = gated_linear_layer(h1_norm, h1_norm_gates)
        
        return out

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
