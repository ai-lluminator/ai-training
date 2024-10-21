import chromadb
import random
from chromadb.utils import embedding_functions
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_user_interest():
    # List of sample AI research interests
    research_interests = [
        "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Robotics",
        "Generative Adversarial Networks", "Explainable AI", "Machine Learning", "Deep Learning",
        "AI Ethics", "Autonomous Systems", "Neural Networks", "Speech Recognition", 
        "AI in Healthcare", "Federated Learning", "Edge AI", "AI in Finance", 
        "Swarm Intelligence", "Quantum Computing for AI", "Multi-Agent Systems", "AI for Cybersecurity",
        "AI-driven Drug Discovery", "Human-AI Interaction", "Transfer Learning", "Self-Supervised Learning",
        "AI for Climate Change", "AI in Education", "AI for Autonomous Vehicles", "Meta-Learning",
        "Causal Inference in AI", "AI for Social Good", "AI in Gaming", "AI and Creativity", 
        "Cognitive Computing", "AI-driven Personalization", "AI for Supply Chain Optimization", 
        "AI in Smart Cities", "AI for Agriculture", "AI in Telecommunications", 
        "AI and Predictive Analytics", "AI for Energy Efficiency", "Ethics of AI in Autonomous Weapons"
    ]

    # List of sample backgrounds
    backgrounds = [
        "is a computer scientist and knows how to program",
        "is in a management position with an MBA",
        "is an AI researcher with a PhD",
        "has a background in data science and machine learning",
        "works as a software engineer",
        "is an entrepreneur in the AI space",
        "has experience in AI-driven business strategies",
        "is an AI ethicist focused on responsible AI development",
        "is a university professor specializing in AI",
        "has experience in robotics and automation"
    ]

    # Function to generate random names
    def generate_random_name():
        first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Quinn", "Drew", "Charlie", "Bailey"]
        last_names = ["Smith", "Johnson", "Lee", "Brown", "Garcia", "Martinez", "Davis", "Lopez", "Hernandez", "Clark"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"

    # Create list of 50 dictionaries with random names, research interests, and backgrounds
    agents_list = [
        {
            "agent name": generate_random_name(),
            "research interest": random.choice(research_interests),
            "background": random.choice(backgrounds)
        }
        for _ in range(50)
    ]
    
    return agents_list

def get_timestep_boundries(collection):
    """
    Retrieve the largest value for the 'timestamp' metadata from a ChromaDB collection.

    Args:
        collection: A ChromaDB collection instance.

    Returns:
        The largest 'timestamp' value, or None if no valid timestamp is found.
    """
    # Fetch all metadata from the collection
    data = collection.get(ids=None, include=['metadatas'])
    
    # Extract metadata
    metadatas = data['metadatas']
    
    # Find the largest timestamp
    largest_timestamp = max(
        (meta.get('published') for meta in metadatas if 'published' in meta),
        default=None
    )

    smallest_timestamp = min(
        (meta.get('published') for meta in metadatas if 'published' in meta),
        default=None
    )
    
    return largest_timestamp, smallest_timestamp

def get_qwen_model():
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="mps"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prompt_interesting_papers(model, tokenizer, documents, interest, background):
    paper_string = ""
    for i, doc in enumerate(documents):
        paper_string += f"{i+1}. {doc}\n"
    messages = [
        {"role": "system", "content": f"You are a scientist who is only answering in plain numbers with a background in {background}. You are interested in {interest}."},
        {"role": "user", "content": f" What is the number of the most interesting paper to you: {paper_string}:"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


def get_chroma_db():
    chroma_client = chromadb.PersistentClient(path="chromadb")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="mps")
    data_collections = chroma_client.get_or_create_collection("arxiv_data", embedding_function=sentence_transformer_ef)
    return data_collections

model, tokenizer = get_qwen_model()

# Connect to your collection
collection = get_chroma_db()

# Get the largest and smallest timestamps in the collection
largest_timestamp, smallest_timestamp = get_timestep_boundries(collection)

agents_list = generate_user_interest()

day_in_seconds = 24 * 60 * 60

results_dict = {user["agent name"]: [] for user in agents_list}

try:
    for day in tqdm(range(smallest_timestamp, largest_timestamp, day_in_seconds), desc="Processing days"):
        # Get the start and end timestamps for the current day
        start_timestamp = day
        end_timestamp = day + day_in_seconds

        interests = [agent['research interest'] for agent in agents_list]

        # Query the collection using metadata filters to get results between these timestamps
        results = collection.query(
            query_texts=interests,
            where={
                "$and": [{"published": {"$gte": start_timestamp}}, {"published":{"$lte": end_timestamp}}]
            },
            n_results=6
        )

        for user, interest, documents in zip(agents_list, interests, results['documents']):
            try:
                response = prompt_interesting_papers(model, tokenizer, documents, interest, user['background'])
                interesting_paper_number = int(response)
                current_dict = {"papers": documents, "interesting paper": interesting_paper_number}
                results_dict[user["agent name"]].append(current_dict)
            except ValueError:
                continue
except KeyboardInterrupt:
    print("Data processing interrupted.")
    # Store the results in a file
    with open("results.json", "w") as f:
        json.dump(results_dict, f, indent=4)