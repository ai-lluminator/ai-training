import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import json

import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import itertools

# Function to find the index of a tensor
def get_tensor_index(tensor_list, target_tensor):
    for idx, tensor in enumerate(tensor_list):
        if torch.equal(tensor, target_tensor):
            return idx
    return -1

class UserPaperDataset(Dataset):
    def __init__(self, user_data, sequence_length=4, batch_size=4, device="mps", embedding_model='all-MiniLM-L6-v2'):
        """
        Args:
            user_data (dict): A dictionary containing users and their papers
        """
        # Load a pre-trained sentence-transformer model for generating embeddings
        self.model = SentenceTransformer(embedding_model)

        self.sequence_length = sequence_length  # Maximum length of a sequence
        self.batch_size = batch_size  # Batch size for processing
        self.device = device  # Device to use for processing
        
        # Store data as a list of sequences per user
        self.user_embeddings = []  # To store embeddings for this user
        self.user_labels = []  # To store labels for this user
        
        for user in user_data:
            sequence_embeddings = []
            sequence_labels = []

            for choice_index in range(len(user_data[user])):
                # Create the embedding and label pairs for this group of paper

                for idx in range(len(user_data[user][choice_index]['papers'])):
                    paper = user_data[user][choice_index]['papers'][idx]
                    current_label = 1 if idx == user_data[user][choice_index]['interesting paper'] - 1 else 0
                    embedding = self.model.encode(paper, convert_to_tensor=True)
                    # Concat embedding and label into one tensor
                    sequence = torch.cat((embedding, torch.tensor([current_label], device=embedding.device)), dim=0)
                    sequence_embeddings.append(sequence)
                    sequence_labels.append(current_label)

            # Append the sequence of embeddings and labels to the user data
            self.user_embeddings.append(sequence_embeddings)
            self.user_labels.append(sequence_labels)

    def embedding_size(self):
        """Returns the size of the embedding vectors."""
        return self.model.encode('test').shape[-1] + 1  # Plus one for the label dimension
        
    def __len__(self):
        # Total number of users in the dataset
        return len(self.user_embeddings) * 5

    def __getitem__(self, idx):
        # Get a random sequence of length self.sequence_length of embeddings from each user
        
        users = torch.randint(0, len(self.user_embeddings), (self.sequence_length,))
        sequences = []
        labels = []
        for user in users:
            sequence = torch.randint(0, len(self.user_embeddings[user]), (self.sequence_length,))
            cur_sequence = torch.stack([self.user_embeddings[user][seq] for seq in sequence])

            # Replace last value of last embedding with -1
            label = cur_sequence[-1, -1].clone()
            cur_sequence[-1, -1] = -1
            sequences.append(cur_sequence)
            labels.append(label)

        batch_inputs = torch.stack(sequences).to(self.device)
        batch_labels = torch.stack(labels).float().to(self.device)
        
        # Return the sequence of embeddings and their corresponding labels
        return batch_inputs, batch_labels