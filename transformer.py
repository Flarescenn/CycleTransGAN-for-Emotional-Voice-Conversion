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


def create_attention_mask_from_input_mask(from_tensor, to_mask):
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
    
    return mask


class AttentionLayer(nn.Module):
    def __init__(self, num_attention_heads=1, size_per_head=1024, dropout_prob=0.0, initializer_range=0.02):
        super(AttentionLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.dropout_prob = dropout_prob
        
        # linear layers for query, key, value
        self.query = nn.Linear(num_attention_heads * size_per_head, num_attention_heads * size_per_head)
        self.key = nn.Linear(num_attention_heads * size_per_head, num_attention_heads * size_per_head)
        self.value = nn.Linear(num_attention_heads * size_per_head, num_attention_heads * size_per_head)

        self.dropout = nn.Dropout(dropout_prob)

        # Initialize weights
        nn.init.normal_(self.query.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.key.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.value.weight, mean=0.0, std=initializer_range)
        
    def transpose_for_scores(self, x, batch_size):
        # Reshaping the input tensor
        x = x.view(batch_size, -1, self.num_attention_heads, self.size_per_head)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_attention_heads, seq_length, size_per_head]

    def forward(self, from_tensor, to_tensor, attention_mask=None):
        batch_size = from_tensor.size(0)
        from_seq_length = from_tensor.size(1)
        to_seq_length = to_tensor.size(1)
        
        # Linear projections for Q, K, V
        query_layer = self.query(from_tensor)
        key_layer = self.key(to_tensor)
        value_layer = self.value(to_tensor)
        
        # Reshape and transpose
        query_layer = self.transpose_for_scores(query_layer, batch_size)
        key_layer = self.transpose_for_scores(key_layer, batch_size)
        
        # Attention scores: QK^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.size_per_head)
        
        # Apply mask if available
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Get the context layer
        value_layer = value_layer.view(batch_size, -1, self.num_attention_heads, self.size_per_head)
        value_layer = value_layer.permute(0, 2, 1, 3)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)
        
        return context_layer


class Transformer(nn.Module):
    def __init__(self, hidden_size=1024, num_hidden_layers=6, num_attention_heads=4,
                 intermediate_size=2048, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 initializer_range=0.02, max_position_embeddings=512, use_position_emb=True):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.use_position_emb = use_position_emb

        # Hidden size must be a multiple of num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})")

        self.attention_head_size = int(hidden_size / num_attention_heads)
        
        # Position embedding
        if use_position_emb:
            self.position_embeddings = nn.Parameter(torch.zeros(max_position_embeddings, hidden_size))
            nn.init.normal_(self.position_embeddings, mean=0.0, std=self.initializer_range)
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerLayer(hidden_size, num_attention_heads, 
                                                      attention_head_size=self.attention_head_size, 
                                                      intermediate_size=intermediate_size,
                                                      hidden_dropout_prob=hidden_dropout_prob,
                                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                      initializer_range=initializer_range) 
                                     for _ in range(num_hidden_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor, attention_mask=None, do_return_all_layers=False):
        """
        Applies the Transformer model on the input tensor with optional positional embeddings and attention mask.

        Args:
            input_tensor: torch.Tensor, shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional torch.Tensor of shape (batch_size, seq_length)
            do_return_all_layers: If True, returns the output of each Transformer layer

        Returns:
            torch.Tensor: Final Transformer output tensor, or all layer outputs if do_return_all_layers is True.
        """
        batch_size, seq_length, _ = input_tensor.size()

        # Add positional embeddings if enabled
        if self.use_position_emb:
            assert seq_length <= self.max_position_embeddings, "Sequence length exceeds max position embeddings"
            position_embeddings = self.position_embeddings[:seq_length, :].unsqueeze(0)
            input_tensor = input_tensor + position_embeddings

        prev_output = self.layer_norm(input_tensor)
        
        all_layer_outputs = []
        
        # Apply each transformer layer
        for layer in self.layers:
            layer_output = layer(prev_output, attention_mask)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if do_return_all_layers:
            return all_layer_outputs
        else:
            return all_layer_outputs[-1]


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size, intermediate_size, 
                 hidden_dropout_prob, attention_probs_dropout_prob, initializer_range):
        super(TransformerLayer, self).__init__()

        # Self-attention
        self.self_attention = AttentionLayer(
            num_attention_heads=num_attention_heads, size_per_head=attention_head_size, 
            dropout_prob=attention_probs_dropout_prob
        )
        self.attention_output_dense = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.attention_output_dense.weight, mean=0.0, std=initializer_range)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Intermediate layer
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        nn.init.normal_(self.intermediate_dense.weight, mean=0.0, std=initializer_range)
        self.intermediate_act_fn = F.gelu

        # Output layer
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        nn.init.normal_(self.output_dense.weight, mean=0.0, std=initializer_range)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, attention_mask=None):
        """
        Performs forward pass through the transformer layer.

        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_length, hidden_size)
            attention_mask: Optional torch.Tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: Output tensor from this transformer layer.
        """
        # Self-attention layer
        attention_output = self.self_attention(hidden_states, hidden_states, attention_mask)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)

        # Intermediate layer
        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)

        # Output layer
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output