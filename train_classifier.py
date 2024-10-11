# Step 1: Install Necessary Libraries
# Make sure to install these libraries before running the script
# !pip install transformers datasets torch

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import re
import tqdm

# Step 1: Define a custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self, keywords_csv, interests_csv, tokenizer, max_length=256):
        # Load the CSV file into a DataFrame
        self.df1 = pd.read_csv(keywords_csv)
        self.df2 = pd.read_csv(interests_csv)

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Step 2: Clean the keywords and create correct pairs
        self.df1['User Input'] = self.df1['Keywords'].apply(self.clean_keywords)
        self.df1['Combined Text'] = self.df1.apply(
            lambda row: f"User Input: {row['User Input']} Science Text: {row['Original Text']}", axis=1
        )

        self.df2['User Input'] = self.df2['Interest']
        self.df2['Combined Text'] = self.df2.apply(
            lambda row: f"User Input: {row['User Input']} Science Text: {row['Original Text']}", axis=1
        )

        self.df = pd.concat([self.df1, self.df2], axis=0, join='outer')

        # Step 3: Generate correct and incorrect combinations
        self.data = []
        
        # Correct combinations with label 0
        for _, row in self.df.iterrows():
            self.data.append((row['Combined Text'], 0))
        
        # Incorrect combinations with label 1
        incorrect_count = 0
        correct_count = len(self.data)
        
        while incorrect_count < 4 * correct_count:
            # Randomly choose two different rows
            original_text_row = self.df.sample(1).iloc[0]
            keywords_row = self.df.sample(1).iloc[0]
            
            # Ensure that we are combining texts from different rows
            if original_text_row.name != keywords_row.name:
                combined_text = f"User Input: {keywords_row['User Input']} Science Text: {original_text_row['Original Text']}"
                self.data.append((combined_text, 1))
                incorrect_count += 1

    def clean_keywords(self, keywords):
        # Remove parentheses and brackets
        cleaned = re.sub(r'[\[\]\(\)]', '', keywords)
        # Remove any extra whitespace
        cleaned = cleaned.strip()
        return cleaned
    
    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get a specific sample
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

# Step 2: Download a Pre-trained LLM from Hugging Face
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 4: Create an instance of the dataset
keywords_csv = '/home/see/Documents/GitHub/AIlluminator/ai/training/database/texts_and_keywords.csv'  # Replace with your CSV file path
interests_csv = '/home/see/Documents/GitHub/AIlluminator/ai/training/database/interests.csv'
full_dataset = CustomTextDataset(keywords_csv, interests_csv, tokenizer)

# Step 5: Split the dataset into training and test sets
train_size = int(0.8 * len(full_dataset))  # 80% for training
test_size = len(full_dataset) - train_size  # 20% for testing

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Step 6: Create DataLoader objects for the training and test sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 4: Define the Training Loop
device = torch.device('cuda')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    with tqdm.tqdm(data_loader, unit="batch") as tepoch:
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tepoch.set_postfix(loss=loss.item())
            tepoch.update(1)

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), model

# Step 5: Define the Evaluation Function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Step 6: Run Training and Evaluation
num_epochs = 1
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    train_acc, train_loss, model = train_epoch(model, train_loader, optimizer, device)
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, test_loader, device)
    print(f'Validation loss {val_loss} accuracy {val_acc}')

torch.save(model.state_dict(), "classifier.pt")
print('Training complete!')
