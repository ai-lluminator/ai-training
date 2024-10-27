import torch
import torch.nn as nn
import math
import json
from regression_dataset import UserPaperDataset
# Import data_loader
from torch.utils.data import random_split, DataLoader
import random
import torch.optim as optim
from progressbar import progressbar

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, dtype=torch.float) * 
                             (-math.log(10000.0) / embedding_dim))  # Shape: [embedding_dim]

        pe = position * div_term  # Shape: [max_len, embedding_dim]
        pe = torch.zeros_like(pe).to(div_term.device)  # Initialize pe with zeros

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term[0::2])

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embedding_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, seq_length, num_heads=11, num_layers=6, dropout=0.0, pos_encoding=False):
        super(TransformerClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.pos_encoding = pos_encoding

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=seq_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc_out = nn.Linear(embedding_dim, 1)  # Output a single logit for binary classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_length, embedding_dim]
        if self.pos_encoding:
            x = self.pos_encoder(x)
        # Transformer expects input shape: [seq_length, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # Revert back to original shape
        x = x.permute(1, 0, 2)
        # Pooling over the sequence dimension (mean pooling)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.fc_out(x).squeeze(-1)  # Output shape: [batch_size]
        return logits


def train(user_data_path, learning_rate, batch_size=8, num_epochs=300, device="mps"):
    # Load data
    user_data = json.load(open(user_data_path, "r"))
    user_paper_dataset = UserPaperDataset(user_data, num_decisions=3)

    # Split dataset into training and validation sets
    train_size = int(0.9 * len(user_paper_dataset))
    val_size = len(user_paper_dataset) - train_size
    train_dataset, val_dataset = random_split(user_paper_dataset, [train_size, val_size])

    # Data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and other components
    embedding_dim = user_paper_dataset.embedding_size()
    model = TransformerClassifier(embedding_dim, seq_length=user_paper_dataset.sequence_length, dropout=0.0).to(device)
    pos_weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {trainable_params}")

    def validate():
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        class_0_correct = 0
        class_0_total = 0
        class_1_correct = 0
        class_1_total = 0

        with torch.no_grad():
            for val_data, val_labels in val_loader:
                logits = model(val_data)
                loss = criterion(logits, val_labels.float())
                val_loss += loss.item()

                preds = (logits >= 0).float()
                val_corrects += (preds == val_labels.float()).sum().item()
                val_samples += val_labels.size(0)

                # Class-specific accuracy update
                for i in range(val_labels.size(0)):
                    if val_labels[i] == 0:
                        class_0_total += 1
                        if preds[i] == val_labels[i]:
                            class_0_correct += 1
                    elif val_labels[i] == 1:
                        class_1_total += 1
                        if preds[i] == val_labels[i]:
                            class_1_correct += 1

        # Validation metrics
        val_loss /= len(val_loader)
        val_acc = val_corrects / val_samples
        class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else 0
        class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else 0

        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Validation Class 0 Accuracy: {class_0_acc:.4f}, Class 1 Accuracy: {class_1_acc:.4f}')

        return val_acc

    # Training loop
    for epoch in progressbar(range(num_epochs), redirect_stdout=True):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0
        class_0_correct = 0
        class_0_total = 0
        class_1_correct = 0
        class_1_total = 0

        for user_data, labels in train_loader:
            logits = model(user_data)
            loss = criterion(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (logits >= 0).float()
            running_corrects += (preds == labels.float()).sum().item()
            running_samples += labels.size(0)

            for i in range(labels.size(0)):
                if labels[i] == 0:
                    class_0_total += 1
                    if preds[i] == labels[i]:
                        class_0_correct += 1
                elif labels[i] == 1:
                    class_1_total += 1
                    if preds[i] == labels[i]:
                        class_1_correct += 1

        # Training metrics
        print("\n\n")
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects / running_samples
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else 0
        class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else 0
        print(f'Class 0 Accuracy: {class_0_acc:.4f}, Class 1 Accuracy: {class_1_acc:.4f}')

        # Validation step
        validate()

        print('---------------------------------')

    return epoch_acc

# Usage example:
accuracies = []
for lr in [0.00005]:
    accuracies.append(train("/Users/cowolff/Documents/GitHub/AI-lluminator/ai-training/transformer-regressor-reranking/results.json", learning_rate=lr, batch_size=8, num_epochs=400, device="mps"))

print(accuracies)