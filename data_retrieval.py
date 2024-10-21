import requests
import xml.etree.ElementTree as ET
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from tqdm import tqdm
import time

def get_arxiv_papers(search_query, start=0, max_results=100):
    # Define the base URL for the arXiv API
    base_url = "http://export.arxiv.org/api/query?"

    # Create the query parameters
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",  # Sort by submission date
        "sortOrder": "descending",  # Newest papers first
        "start": start,  # The index of the first result you want to retrieve
        "max_results": max_results  # The number of results you want to retrieve
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: Unable to retrieve data (status code {response.status_code})")
        return []

    # Parse the response XML
    root = ET.fromstring(response.content)

    # Extract relevant data from each entry
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        print(entry)
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        published = entry.find("{http://www.w3.org/2005/Atom}published").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
        paper_url = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()

        papers.append({
            "title": title,
            "published": published,
            "summary": summary,
            "authors": authors,
            "url": paper_url
        })

    return papers

def get_chroma_db():
    chroma_client = chromadb.PersistentClient(path="chromadb_new")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="mps")
    data_collections = chroma_client.get_or_create_collection("arxiv_data", embedding_function=sentence_transformer_ef)
    return data_collections

number_samples = 1000000
step_size = 500

for i in tqdm(range(25 * step_size, number_samples, step_size)):
    while True:
        try:
            time.sleep(4)
            search_query = "all:artificial intelligence"
            papers = get_arxiv_papers(search_query, start=i, max_results=step_size)

            documents = [f"{paper['title']}\n{paper['summary']}" for paper in papers]
            ids = [paper['url'] for paper in papers]

            metadata = [{"title": paper['title'], "published": int(datetime.strptime(paper['published'], "%Y-%m-%dT%H:%M:%SZ").timestamp()), "authors": ', '.join(paper['authors'])} for paper in papers]

            data_collections = get_chroma_db()
            data_collections.add(documents=documents, ids=ids, metadatas=metadata)

            break
        except Exception as e:
            print(e)
            continue