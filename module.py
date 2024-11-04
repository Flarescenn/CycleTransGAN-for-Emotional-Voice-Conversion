import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import *
activation_fuc = F.leaky_relu

#please note: input here is number of in-channels, and filters is the number of out-channels

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
    def __init__(self, num_features, epsilon=1e-06, activation=None, name = None, is_2d=True):
        super(instance_norm_layer, self).__init__()
        if is_2d:
            self.instance_norm = nn.InstanceNorm2d(num_features, eps=epsilon, affine=False)
        else:
            self.instance_norm = nn.InstanceNorm1d(num_features, eps=epsilon, affine=False)
        self.activation = activation

    def forward(self, inputs):
        x = self.instance_norm(inputs)
        if self.activation:
            x = activation_fuc(x)
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
    Tensor of shape (N, C, W) ---> Tensor of Shape (N, C/r, W*r)
    For upsampling.

    Args: 
        shuffle_size (int): The factor by which to increase the temporal dimension
    Shape:
        - Input: (N, C, W) where
            * N is the batch size
            * W is the width
            * C is the number of channels (must be divisible by shuffle_size)
        - Output: (N, C/shuffle_size, W*shuffle_size)
    """
    def __init__(self, shuffle_size=2):
        super(pixel_shuffler, self).__init__()
        self.shuffle_size = shuffle_size
    
    def forward(self, x):
        N = x.size(0)
        W = x.size(2)
        C = x.size(1)

        out_channels = C// self.shuffle_size
        out_width = W * self.shuffle_size

        return x.reshape(N, out_channels, out_width)


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
            num_features=filters,
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
            num_features=filters,
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
            num_features=filters // 2,
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
            num_features=filters,
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
            num_features=filters, 
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
            num_features=filters // shuffle_size,
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
            num_features=filters // shuffle_size,
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


class downsample2d_block(nn.Module):
    """
    2D downsampling block 
    Architecture:
        Input → (Conv2D → InstanceNorm2D) ⨁ (Conv2D → InstanceNorm2D) → GLU
    """
    def __init__(self, inputs, filters, kernel_size, strides):
        super(downsample2d_block, self).__init__()
        #for both the paths as if the code isnt already long enough
        self.conv2d = conv2d_layer( 
            inputs= inputs,
            filters = filters,
            kernel_size = kernel_size, 
            strides = strides, 
            activation=True
        )
        self.norm2d = instance_norm_layer(
            num_features=filters,
            activation=True,
            is_2d=True
        )

    def forward(self, x):
        # Main Path
        h1 = self.conv2d(x)
        h1_norm = self.norm2d(h1)

        # Gating Path
        h1_gates = self.conv2d(x)
        h1_norm_gates = self.norm2d(h1_gates)

        # Apply GLU
        out = gated_linear_layer(h1_norm, h1_norm_gates)

        return out



class generator_gatedcnn(nn.Module):
    def __init__(self, inputs):
        super(generator_gatedcnn, self).__init__()
        # Generator layers defined here

        self.input_transpose = lambda x: x.permute(0, 2, 1)
        self.h1_conv = conv1d_layer(inputs, filters=128, kernel_size=15, strides=1, activation=True)
        self.h1_conv_gates = conv1d_layer(inputs, filters=128, kernel_size=15, strides=1, activation=True)
        
        self.downsample1 = downsample1d_block(128, filters=256, kernel_size=5, strides=2)
        self.downsample2 = downsample1d_block(256, filters=512, kernel_size=5, strides=2)

        self.residual1 = residual1d_block(512, filters=1024, kernel_size=3, strides=1) #returns 512 channels
        #to implement
        
        self.transformer1 = Transformer(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1024
        )      #returns 512

        self.residual2 = residual1d_block(512, filters=1024, kernel_size=3, strides=1)  #returns 512
        self.transformer2 = Transformer(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=16,
            intermediate_size=2048,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1024
        )       #returns 512

        #in channels not calculated yet
        self.upsample1 = upsample1d_block(512, filters=1024, kernel_size=5, strides=1, shuffle_size=2)
        self.upsample2 = upsample1d_block(512, filters=512, kernel_size=5, strides=1, shuffle_size=2)

        self.output_conv = conv1d_layer(256, filters=24, kernel_size=15, strides=1)
        self.output_transpose = lambda x: x.permute(0, 2, 1)

    def forward(self, inputs):
        
        #x = self.input_transpose(inputs)
        x = inputs
        batch_size, seq_length, _ = x.shape
        #input_mask = torch.ones(batch_size, seq_length, dtype=torch.int32, device=x.device)
        #ttention_mask = create_attention_mask(x, input_mask)

        # First conv and gated linear unit
        h1 = self.h1_conv(x)
        h1_gates = self.h1_conv_gates(x)
        h1_glu = gated_linear_layer(h1, h1_gates)

        # Downsampling
        d1 = self.downsample1(h1_glu)
        d2 = self.downsample2(d1)
        
        # Residual blocks with transformers
        r1 = self.residual1(d2)
        r1_transformed = r1.transpose(1, 2) #[B,L,C]
        batch_size, seq_length, _ = r1_transformed.shape
        input_mask = torch.ones(batch_size, seq_length, device=r1_transformed.device)
        attention_mask = create_attention_mask(r1_transformed, input_mask)
        t1 = self.transformer1(r1_transformed, attention_mask)
        t1 = t1.transpose(1, 2) # [B, C, L]

        r2 = self.residual2(r1)

        r2_transformed = r2.transpose(1, 2)
        batch_size, seq_length, _ = r2_transformed.shape
        input_mask = torch.ones(batch_size, seq_length, device=r2_transformed.device)
        attention_mask = create_attention_mask(r2_transformed, input_mask)

        t2 = self.transformer2(r2_transformed, attention_mask)
        t2 = t2.transpose(1,2)

        # Upsampling
        u1 = self.upsample1(t2)
        u2 = self.upsample2(u1)
        
        # Final convolution and output transpose
        o1 = self.output_conv(u2)
        #o2 = self.output_transpose(o1)
        
        return o1


class discriminator(nn.Module):
    def __init__(self, inputs):
        super(discriminator, self).__init__()
        
        self.gp_weight = 10.0
        #self.instance_norm = nn.InstanceNorm2d(128)
        self.input_expand = lambda x: x.unsqueeze(2)
        #self.noise_std = 0.01
        self.h1_conv = conv2d_layer(inputs, filters=128, kernel_size=(3, 3), strides=(1, 2), activation=True)
        self.h1_conv_gates = conv2d_layer(inputs, filters=128, kernel_size=(3, 3), strides=(1, 2), activation=True)
        
        # Downsample blocks
        self.downsample1 = downsample2d_block(128, filters=256, kernel_size=(3, 3), strides=(2, 2))
        self.downsample2 = downsample2d_block(256, filters=512, kernel_size=(3, 3), strides=(2, 2))
        self.downsample3 = downsample2d_block(512, filters=1024, kernel_size=(3, 4), strides=(1, 8))
        
        # Output dense layer
        self.output_dense = nn.Linear(2048, 1)
        # self.output_dense = nn.Linear(8192, 1)
        #self.output_activation = nn.Sigmoid()
        

    def forward(self, inputs):

        x = self.input_expand(inputs)        
        h1 = self.h1_conv(x)      
        h1_gates = self.h1_conv_gates(x)
        h1_glu = gated_linear_layer(h1, h1_gates)
        d1 = self.downsample1(h1_glu)
        
        d2 = self.downsample2(d1)        
        d3 = self.downsample3(d2)

        d3_flat = d3.view(d3.size(0), -1) 
        
        o1 = self.output_dense(d3_flat)
        #o1 = self.output_activation(o1)
        #print(f"Final output shape: {o1.shape}")

        return o1

