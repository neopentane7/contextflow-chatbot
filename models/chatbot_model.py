import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class HybridChatbotModel(nn.Module):
    """Hybrid LSTM-Transformer model for conversational AI"""
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, 
                 num_heads=8, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer for sequential processing
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Adjust dimension for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_output_dim,
            nhead=num_heads,
            dim_feedforward=lstm_output_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(lstm_output_dim, num_heads)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(lstm_output_dim, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
    def forward(self, x, mask=None):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(lstm_out, src_key_padding_mask=mask)
        
        # Apply attention
        attended_out = self.attention(transformer_out, transformer_out, transformer_out, mask)
        
        # Dropout and output projection
        output = self.dropout(attended_out)
        output = self.output_layer(output)  # (batch, seq_len, vocab_size)
        
        return output


class ContextManager:
    """Manages conversation context across multiple turns"""
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        
    def update_context(self, user_input, bot_response):
        from datetime import datetime
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'timestamp': datetime.now()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_context_string(self):
        context_parts = []
        for turn in self.conversation_history:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Bot: {turn['bot']}")
        return " ".join(context_parts)
    
    def clear_context(self):
        self.conversation_history = []


# Model configuration
MODEL_CONFIG = {
    'vocab_size': 10000,  # Will be updated based on actual vocabulary
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 3,
    'num_heads': 8,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}
