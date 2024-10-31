import kagglehub
import json
from chromadb.utils import embedding_functions
import chromadb
from tqdm import tqdm
from datetime import datetime
import time

def load_data(filepath, limit=None):
    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data

def get_chroma_db():
    chroma_client = chromadb.PersistentClient(path="chromadb")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda")
    data_collections = chroma_client.get_or_create_collection("arxiv_data", embedding_function=sentence_transformer_ef)
    return data_collections

# Download latest version
path = kagglehub.dataset_download("Cornell-University/arxiv")

json_file = f"{path}/arxiv-metadata-oai-snapshot.json"

# Load the data
data = load_data(json_file)

# Connect to your collection
collection = get_chroma_db()

print("Creating data for chroma db")
documents = [f"{paper['title']}\n{paper['abstract']}" for paper in data]
ids = [paper['id'] for paper in data]
metadata = [{"title": paper['title'],
             "published": int(time.mktime(datetime.strptime(paper['versions'][-1]['created'], "%a, %d %b %Y %H:%M:%S %Z").timetuple())),
             "authors": ', '.join([' '.join(author_data) for author_data in paper['authors_parsed']])}
             for paper in data]

step_size = 1000

for i in tqdm(range(0, len(documents), step_size)):
    while True:
        try:
            cur_documents = documents[i:i + step_size]
            cur_ids = ids[i:i + step_size]
            cur_metadata = metadata[i:i + step_size]

            data_collections = get_chroma_db()
            data_collections.add(documents=cur_documents, ids=cur_ids, metadatas=cur_metadata)

            break
        except Exception as e:
            print(e)
            break
