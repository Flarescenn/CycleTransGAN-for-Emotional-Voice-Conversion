import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import six

# GELU activation
def gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    return F.dropout(input_tensor, p=dropout_prob, training=True)


def layer_norm(input_tensor, name=None):
    layer_norm = nn.LayerNorm(input_tensor.size(-1), eps=1e-12)  #eps = 1e-12 for precision
    if torch.cuda.is_available():
        layer_norm = layer_norm.cuda()
    return layer_norm(input_tensor)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """
    Applies layer normalization followed by dropout to the input tensor.
    
    Args:
    - input_tensor (torch.Tensor): The tensor to normalize and apply dropout to.
    - dropout_prob (float): Probability of an element to be zeroed.
    - normalized_shape (int or list of int): Shape for the layer normalization (usually the last dimension size).
    - name (str, optional): Name for the layer (optional for PyTorch).
    
    Returns:
    - torch.Tensor: The normalized and dropout-applied tensor.
    """
    # Apply layer normalization
    output_tensor = layer_norm(input_tensor, name)
    # Apply dropout
    output_tensor = dropout(output_tensor, dropout_prob)

    return output_tensor


def assert_rank(tensor, expected_rank, name=None):
    """
    Raises ValueError if the tensor rank is not of expected rank. (For validating the rank)
    """
    if name is None:
        name = "tensor"
        
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
            
    actual_rank = len(tensor.size())
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            f"For the tensor `{name}`, the actual rank "
            f"`{actual_rank}` (shape = {tuple(tensor.size())}) "
            f"is not equal to the expected rank `{expected_rank}`")
    

def get_shape_list(tensor, expected_rank=None, name=None):
    """
    Returns a list of dimensions of the tensor.
    """
    if name is None:
        name = "tensor" 
        
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = list(tensor.size())
    return shape


def create_initializer(initializer_range=0.02):
    """
    Creates a truncated normal initializer.
    """
    def initializer(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    return initializer


def reshape_to_matrix(input_tensor):
    """
    Reshapes input tensor to a matrix (2D tensor).
    """
    ndims = len(input_tensor.size())
    if ndims < 2:
        raise ValueError(f"Input tensor must have at least rank 2. Shape = {input_tensor.size()}")
    if ndims == 2:
        return input_tensor
        
    width = input_tensor.size(-1)
    output_tensor = input_tensor.view(-1, width)
    return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    """
    Reshapes a matrix back to its original shape.
    """
    if len(orig_shape_list) == 2:
        return output_tensor
        
    width = output_tensor.size(-1)
    return output_tensor.view(orig_shape_list[:-1] + [width])


'''def create_attention_mask(from_tensor, to_mask):
        """
        Creates 3D attention mask from a 2D tensor mask.
        """
        from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        
        to_shape = get_shape_list(to_mask, expected_rank=2)
        to_seq_length = to_shape[1]
        
        # Convert mask to float and add a dimension
        to_mask = to_mask.float().view(batch_size, 1, to_seq_length)
        
        # Create broadcast ones
        broadcast_ones = torch.ones(
            batch_size, from_seq_length, 1,
            dtype=torch.float,
            device=from_tensor.device)
            
        # Create the attention mask
        mask = broadcast_ones * to_mask
        
        return mask'''
def create_attention_mask(from_tensor, to_mask):
    
    batch_size, from_seq_length, _ = from_tensor.size()
    to_seq_length = to_mask.size(1)
    mask = to_mask.unsqueeze(1).expand(batch_size, from_seq_length, to_seq_length)
    
    return mask

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Create query, key, value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):

        batch_size, seq_length, _ = x.size()
        new_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()
        # Linear projections for Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Reshape and transpose
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        #print(f"Attention scores shape: {attention_scores.shape}")
        if attention_mask is not None:
            #print(f"Attention mask shape: {attention_mask.shape}")

            # [batch_size, 1, 1, seq_length] or [batch_size, 1, seq_length, seq_length]
            if len(attention_mask.shape) == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            #print(f"Reshaped attention mask shape: {attention_mask.shape}")
            
            # Make sure mask broadcasts correctly
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0,
                -1e9
                #-1000.0
            )
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Get the context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        output = self.output(context_layer)

        return output


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size=2048, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512):
        super().__init__()

        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_position_embeddings, hidden_size)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob
            )
            for _ in range(num_hidden_layers)
        ])
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)


    def forward(self, hidden_states, attention_mask=None):
        seq_length = hidden_states.size(1)
        position_embeddings = self.position_embeddings[:, :seq_length, :]
        hidden_states = hidden_states + position_embeddings

        # Apply initial dropout and layer norm
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads,  
                 hidden_dropout_prob, attention_probs_dropout_prob):
        super().__init__()

        # Self-attention
        self.attention = AttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Intermediate layer
        self.intermediate = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU()
        )

        # Output layer
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        """
        Performs forward pass through the transformer layer.

        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional torch.Tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Output tensor from this transformer layer.
        """
        # Self-attention block
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.norm1(attention_output + hidden_states)

        # Feed-forward block
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.norm2(layer_output + attention_output)

        return layer_output
