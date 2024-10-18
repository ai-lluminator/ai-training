import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import json

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

class UserPaperDataset(Dataset):
    def __init__(self, user_data, num_papers=3):
        """
        Args:
            user_data (dict): A dictionary containing users and their papers
        """
        # Load a pre-trained sentence-transformer model for generating embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Store data as a list of sequences per user
        self.user_embeddings = []  # To store embeddings for this user
        self.user_labels = []  # To store labels for this user
        
        for user in user_data:

            for start_idx in range(0, len(user_data[user]) - (num_papers + 1)):
                # Create the embedding and label pairs for this group of papers
                sequence_embeddings = []

                for idx in range(num_papers):
                    index = user_data[user][start_idx + idx]['interesting paper'] - 1
                    paper = user_data[user][start_idx + idx]['papers'][index]
                    embedding = self.model.encode(paper, convert_to_tensor=True)
                    sequence_embeddings.append(embedding)
                    sequence_embeddings.append(torch.tensor([1], device="mps"))

                for predict_idx in range(len(user_data[user][start_idx + num_papers]['papers'])):
                    predict_paper = user_data[user][start_idx + num_papers]['papers'][predict_idx]
                    predict_embedding = self.model.encode(predict_paper, convert_to_tensor=True)
                    current_data = sequence_embeddings + [predict_embedding]
                    current_data = torch.tensor(current_data)
                    current_label = 1 if predict_idx == user_data[user][start_idx + num_papers]['interesting paper'] - 1 else 0
                    self.user_embeddings.append(current_data)
                    self.user_labels.append(current_label)
        
    def __len__(self):
        # Total number of users in the dataset
        return len(self.user_embeddings)

    def __getitem__(self, idx):
        # Retrieve the precomputed embeddings and labels for the user at index `idx`
        input_data = self.user_embeddings[idx]
        label = self.user_labels[idx]
        
        # Return the sequence of embeddings and their corresponding labels
        return input_data, label