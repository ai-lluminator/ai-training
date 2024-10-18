import torch
import torch.nn as nn
import math
import json
from regression_dataset import UserPaperDataset

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.seq_length = 1539
        self.embedding_dim = embedding_dim

        # Positional encoding to retain sequence information
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=self.seq_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer for binary classification
        self.fc_out = nn.Linear(embedding_dim, 1)  # Output a single logit for binary classification

    def forward(self, x):
        # x shape: [batch_size, seq_length, embedding_dim]

        # Add positional encoding
        x = self.positional_encoding(x)  # Shape: [batch_size, seq_length, embedding_dim]

        # Rearrange dimensions to match Transformer input requirements
        x = x.permute(1, 0, 2)  # Shape: [seq_length, batch_size, embedding_dim]

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: [seq_length, batch_size, embedding_dim]

        # Global average pooling over the sequence length
        x = x.mean(dim=0)  # Shape: [batch_size, embedding_dim]

        # Output layer
        logits = self.fc_out(x).squeeze(-1)  # Shape: [batch_size]

        return logits  # For binary classification, use BCEWithLogitsLoss during training

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1539):
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sine and cosine functions to even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x


user_data = json.load(open("results.json", "r"))
user_paper_dataset = UserPaperDataset(user_data)

# Get input shape from the dataset
d_model = user_paper_dataset[0][0].shape
print("Input shape:", d_model)
d_model = d_model[0]  # Get the embedding dimension

# Initialize the model
model = TransformerPredictor(d_model=d_model)
for parameter in model.parameters():
    print(parameter)

batch_size = 16

# Example input data (batch_size x input_length x d_model)
input_data = torch.rand(batch_size, d_model)  # Pre-embedded input data

# Forward pass
output = model(input_data)

# Output shape should be [batch_size, 1] (predicting one token)
print("Output shape:", output.shape)
