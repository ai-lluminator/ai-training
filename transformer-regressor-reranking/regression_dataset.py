import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import json

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import itertools
import copy

# Function to find the index of a tensor
def get_tensor_index(tensor_list, target_tensor):
    for idx, tensor in enumerate(tensor_list):
        if torch.equal(tensor, target_tensor):
            return idx
    return -1

class UserPaperDataset(Dataset):
    def __init__(self, user_data, num_decisions=4, device="mps", embedding_model='all-MiniLM-L6-v2'):
        """
        Args:
            user_data (dict): A dictionary containing users and their papers
        """
        # Load a pre-trained sentence-transformer model for generating embeddings
        self.model = SentenceTransformer(embedding_model)

        self.num_decisions = num_decisions  # Maximum length of a sequence
        self.device = device  # Device to use for processing
        
        # Store data as a list of sequences per user
        self.user_embeddings = []  # To store embeddings for this user
        self.user_labels = []  # To store labels for this user
        
        for user in user_data:

            for choice_index in range(len(user_data[user]) - num_decisions):
                # Create the embedding and label pairs for this group of paper

                sequence_embeddings = []
                for i in range(num_decisions - 1):
                    for idx in range(len(user_data[user][choice_index + i]['papers'])):
                        paper = user_data[user][choice_index + i]['papers'][idx]
                        current_label = 1 if idx == user_data[user][choice_index + i]['interesting paper'] - 1 else 0
                        embedding = self.model.encode(paper, convert_to_tensor=True)
                        # Concat embedding and label into one tensor
                        sequence = torch.cat((embedding, torch.tensor([current_label], device=embedding.device)), dim=0)
                        sequence_embeddings.append(sequence)

                for test_index in range(len(user_data[user][choice_index + num_decisions - 1]['papers'])):
                    paper = user_data[user][choice_index + num_decisions - 1]['papers'][test_index]
                    current_label = 1 if test_index == user_data[user][choice_index + num_decisions - 1]['interesting paper'] - 1 else 0
                    embedding = self.model.encode(paper, convert_to_tensor=True)
                    sequence = torch.cat((embedding, torch.tensor([-1], device=embedding.device)), dim=0)

                    # Deep copy sequence_length and append sequence to it
                    new_sequence = copy.deepcopy(sequence_embeddings)
                    new_sequence.append(sequence)
                    new_sequence = torch.stack(new_sequence)
                    self.user_embeddings.append(new_sequence)
                    self.user_labels.append(torch.tensor(current_label, dtype=torch.long, device=sequence.device))

        self.sequence_length = len(self.user_embeddings[0])

    def embedding_size(self):
        """Returns the size of the embedding vectors."""
        return self.model.encode('test').shape[-1] + 1  # Plus one for the label dimension
        
    def __len__(self):
        # Total number of users in the dataset
        return len(self.user_embeddings)

    def __getitem__(self, idx):
        # Get a random sequence of length self.sequence_length of embeddings from each user
        inputs = self.user_embeddings[idx]
        labels = self.user_labels[idx]
        
        # Return the sequence of embeddings and their corresponding labels
        return inputs, labels