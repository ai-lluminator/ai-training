# Assuming you have a function `generate(prompt)` available and ChromaDB already set up.

# Step 1: Set up your ChromaDB client and retrieve all entries
import chromadb
import csv
from mlx_lm import load, generate
import tqdm

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

def generate_answer(prompt):
    """
    Generate keywords based on the provided prompt using the pre-trained language model.
    
    Parameters:
    - prompt (str): The prompt to generate keywords from.
    
    Returns:
    - keywords (str): The generated keywords.
    """
    keywords = generate(model, tokenizer, prompt=prompt, repetition_penalty=1.0, max_tokens=40)
    return keywords

# Initialize the ChromaDB client (assume default settings or modify as needed)
chroma_client = chromadb.PersistentClient(path="/Users/cowolff/Documents/GitHub/AIlluminator/ai/training/chromadb")

# Specify the collection from which you want to extract entries
collection_name = "Green-AI"  # replace with your actual collection name
collection = chroma_client.get_collection(collection_name)

# Retrieve all documents/entries from the specified collection
entries = collection.get()

# Step 2: Iterate over all samples and use `generate` function to extract keywords
def extract_interest_from_entries(entries):
    interests_dict = {}  # Dictionary to store entry_id: keywords pairs

    stop_step = 10

    # Initialize progress bar
    with tqdm.tqdm(total=len(entries['ids']), desc="Extracting Keywords") as pbar:

        # Open csv file to write the data
        with open('/Users/cowolff/Documents/GitHub/AIlluminator/ai/training/database/interests.csv', mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Entry ID', 'Original Text', 'Interest']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for id, entry in zip(entries['ids'], entries['documents']):
                
                # Step 3: Use the `generate` function to extract keywords
                prompt = f"<|start_header_id|>system<|end_header_id|>You are a scientist.<|eot_id|><|start_header_id|>user<|end_header_id|> Express your general (like medicine, ai, neuroscience etc.) and specific research interest that fits the main topic of the following text, without explaining it and in one continuous sentence: {entry}<|eot_id|><|start_header_id|>assistant<|end_header_id|>Here is your research interest: "
                interest = generate_answer(prompt)

                # Store the result in the dictionary
                interests_dict[id] = interest
                writer.writerow({'Entry ID': id, 'Original Text': entry, 'Interest': interest})
                pbar.update(1)

    return interests_dict

# Step 4: Run the function and print the results
keywords_extracted = extract_interest_from_entries(entries)