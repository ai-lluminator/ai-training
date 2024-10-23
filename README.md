# AI components of the AI/lluminator

This repository contains two separate projects, each designed for different purposes related to research paper recommendation systems. The projects are located in separate folders, and each folder contains the scripts required to run the respective tasks.

## 1. Fine-Tuning the Embedding Model

This project focuses on generating datasets and fine-tuning an embedding model to create domain-specific embeddings based on user interests or keywords.

### Key Features:
- **Dataset Generation**: Scripts to generate synthetic datasets using LLMs (`generate_interest_dataset.py` and `generate_keyword_dataset.py`).
- **Model Fine-Tuning**: Script for fine-tuning a pre-trained embedding model (`train_embedding.py`).
- **Usage**: The resulting embeddings can be used for tasks like similarity search, classification, or clustering.

Find the details in the [/embedding](https://github.com/ai-lluminator/ai-training/tree/main/embedding) folder.

## 2. Predicting User Clicks on Forwarded Papers

This project aims to predict whether a user will click on a forwarded research paper, helping optimize the presentation of relevant papers based on user interaction history.

### Key Features:
- **Click Prediction Model**: A regression model predicts the probability of a user clicking on a forwarded paper.
- **Re-Ranking Papers**: The system re-ranks the list of forwarded papers based on predicted click probabilities to show the most relevant papers.
- **Usage**: The model uses user interaction data and features from the papers for predictions.

Find the details in the [/transformer-regressor-reranking](https://github.com/ai-lluminator/ai-training/tree/main/transformer-regressor-reranking) folder.

### Licence:

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). You are free to use, modify, and distribute this software, but any derivative works must also be open-sourced under the same license.

For more details, please refer to the License file.