import json
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
import pandas as pd
import re
from sentence_transformers import losses

BATCH_SIZE=8
EPOCHS = 2

model_id = "BAAI/bge-small-en"
model = SentenceTransformer(model_id)

def clean_keywords(keywords):
    # Remove parentheses and brackets
    cleaned = re.sub(r'[\[\]\(\)]', '', keywords)
    # Remove any extra whitespace
    cleaned = cleaned.strip()
    return cleaned

df2 = pd.read_csv('/Users/cowolff/Documents/GitHub/AIlluminator/ai/training/database/interests.csv')
df1 = pd.read_csv('/Users/cowolff/Documents/GitHub/AIlluminator/ai/training/database/texts_and_keywords.csv')

# Step 2: Clean the keywords and create correct pairs
df1['User Input'] = df1['Keywords'].apply(clean_keywords)
df1['Combined Text'] = df1.apply(
    lambda row: f"User Input: {row['User Input']} Science Text: {row['Original Text']}", axis=1
)

df2['User Input'] = df2['Interest']
df2['Combined Text'] = df2.apply(
    lambda row: f"User Input: {row['User Input']} Science Text: {row['Original Text']}", axis=1
)

df = pd.concat([df1, df2], axis=0, join='outer')

samples = []
for _, row in df.iterrows():
    samples.append(InputExample(texts=[row['User Input'], row['Original Text']], label=0))

loader = DataLoader(
    samples, batch_size=BATCH_SIZE
)

loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = int(len(loader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='exp_finetune',
    show_progress_bar=True,
    evaluation_steps=50,
)