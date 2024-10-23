### Fine-Tuning the Embedding Model for the AI/lluminator

This repository contains scripts for generating datasets and fine-tuning an embedding model. The main focus is on creating keyword or interest-based datasets and using them to fine-tune a pre-trained embedding model.

### Workflow

1. **Generate Synthetic Dataset:**
   - Use `generate_interest_dataset.py` or `generate_keyword_dataset.py` to create datasets for training. These scripts process raw paper data and use LLMs to create interest strings based on the data.

2. **Fine-Tune Embedding Model:**
   - Run `train_embedding.py` to fine-tune an existing embedding model using the generated dataset. The model learns domain-specific embeddings for keywords or interests.

3. **Evaluate and Use:**
   - After fine-tuning, the embeddings can be used for tasks like similarity search, classification, or clustering.