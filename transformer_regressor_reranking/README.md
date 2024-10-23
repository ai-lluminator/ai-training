## Predicting User Clicks on Forwarded Papers for the AI/lluminator

WORK IN PROGRESS: This folder contains the training scrips for predicting whether a user will click on a forwarded research paper, transforming this interaction into a regression problem. The model leverages historical data on user interactions with forwarded papers to make predictions about future preferences, helping to optimize the presentation of relevant papers. This method is inspired by the paper ["Predicting from Strings: Language Model Embeddings for Bayesian Optimization"](https://arxiv.org/pdf/2410.10190).

### Overview

This approach focuses on predicting the likelihood of a user clicking on a forwarded paper based on their interaction history. By modeling the probability of a click as a continuous variable, the system can more effectively re-rank papers, ensuring that the most relevant ones are shown to the user.

### Key Components

1. **Click-based Data for Regression**: 
   The model uses the user's history of interactions with forwarded papers as input. The target variable is whether the next paper forwarded to the user will be clicked.

2. **Regression Model**: 
   The regression problem is formulated using features from the forwarded paper, the user’s query, and their interaction history. The model learns to predict the probability that the user will click on the forwarded paper based on these features.

3. **Re-Ranking Process**: 
   After predicting the click probability for each paper in the list of forwarded papers, the results are re-ranked based on these predictions to prioritize papers that are more likely to be clicked.

### Method

1. **Feature Representation**: 
    - The model combines several types of features:
      - **Paper Features**: Embeddings of the paper’s abstract, title, and keywords.
      - **User Features**: Past interactions such as clicked papers and search queries.
      - **Interaction Features**: Which papers have been clicked by the user and recency of past clicks.

2. **Training**: 
    - A transformer based regression model is trained using historical data on forwarded papers and user interactions. 
    - The model minimizes a loss function to predict the likelihood of a user clicking on a paper.

3. **Inference and Re-Ranking**:
    - During inference, the model computes the predicted probability of a user clicking on each forwarded paper.
    - The papers are then re-ranked based on these probabilities, presenting the user with the most relevant papers at the top of the list.

4. **Predictive Features**: 
    - **Probability of clicking the paper**: The higher the outputed probability is for the user to click the paper, the higher it gets ranked.

### Usage

1. **Input Data**: 
    - The model expects data on forwarded research papers and the user’s interaction history. This data includes the text of the papers, user queries, and click records.

2. **Training**: 
    - Train the model using historical interaction data. Adjust model hyperparameters to improve prediction performance.

3. **Re-Ranking**: 
    - After training, the model can predict the likelihood of a user clicking on a paper and re-rank the list of forwarded papers based on these predictions.

### References

This method is inspired by the paper "Predicting from Strings: Language Model Embeddings for Bayesian Optimization". It applies a similar concept of embedding strings and predicting outcomes, but here for the purpose of re-ranking search results in response to user queries.