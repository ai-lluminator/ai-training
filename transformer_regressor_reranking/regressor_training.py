import torch
import torch.nn as nn
import math
import json
from regression_dataset import UserPaperDataset
import random
import torch.optim as optim

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
    def __init__(self, embedding_dim, seq_length, num_heads=5, num_layers=4, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

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


user_data = json.load(open("/Users/cowolff/Documents/GitHub/AI-lluminator/ai-training/transformer_regressor_reranking/results.json", "r"))
user_paper_dataset = UserPaperDataset(user_data, sequence_length=6)

batch_size = 4
num_epochs = 150
learning_rate = 0.001

embedding_dim = user_paper_dataset.embedding_size()

print(len(user_paper_dataset))
print(user_paper_dataset[0][0].shape)

exit()

model = TransformerClassifier(embedding_dim, seq_length=user_paper_dataset.sequence_length).to("mps")
pos_weight = torch.tensor([5.0], device="mps")
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # Decrease LR by factor 0.1 every 10 epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0

    class_0_correct = 0
    class_0_total = 0
    class_1_correct = 0
    class_1_total = 0

    for batch_idx in range(len(user_paper_dataset)):
        user_data, labels = user_paper_dataset[batch_idx]
        logits = model(user_data).squeeze()  # Ensure logits have the correct shape
        loss = criterion(logits, labels.float())  # Convert labels to float if necessary

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute predictions for binary classification
        preds = (logits >= 0).float()  # Threshold logits at zero

        # Update correct predictions
        corrects = torch.sum(preds == labels.float()).item()
        running_corrects += corrects
        running_samples += labels.size(0)

        # Class-specific accuracy update
        for i in range(labels.size(0)):
            if labels[i] == 0:  # Class 0
                class_0_total += 1
                if preds[i] == labels[i]:
                    class_0_correct += 1
            elif labels[i] == 1:  # Class 1
                class_1_total += 1
                if preds[i] == labels[i]:
                    class_1_correct += 1

    epoch_loss = running_loss / 10
    epoch_acc = running_corrects / running_samples
    print("-----------")
    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(user_paper_dataset)}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0

    class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else 0
    class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else 0
    print(f'Class 0 Accuracy: {class_0_acc:.4f}, Class 1 Accuracy: {class_1_acc:.4f}')

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}], Learning Rate: {current_lr:.6f}")
    print("-----------\n\n")