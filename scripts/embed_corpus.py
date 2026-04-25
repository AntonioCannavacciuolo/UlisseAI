import os
from dotenv import load_dotenv

# Load .env first
load_dotenv()

# Force HuggingFace to use a local cache folder to avoid WinError 3 on missing drives
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

def get_path_from_env(env_var, default_folder):
    """Get path from env, fallback to project_root / default_folder."""
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / default_folder

def main():
    # 1. Load env vars
    load_dotenv()
    
    # 2. Determine paths
    vectordb_dir = get_path_from_env("VECTORDB_PATH", "vectordb")
    
    corpus_clean_path = vectordb_dir / "corpus_clean.json"
    chroma_dir = vectordb_dir / "chroma"
    
    if not corpus_clean_path.exists():
        print(f"Error: {corpus_clean_path} not found.")
        print("Please provide a valid corpus_clean.json file.")
        return
        
    print("Loading corpus_clean.json...")
    with open(corpus_clean_path, "r", encoding="utf-8") as f:
        try:
            chunks = json.load(f)
        except json.JSONDecodeError:
            print("Error parsing corpus_clean.json. Ensure it is valid JSON.")
            return

    # 3. Initialize ChromaDB client
    print(f"Initializing ChromaDB at {chroma_dir}...")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # 4. Setup embedding function
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="ulisse_brain",
        embedding_function=sentence_transformer_ef
    )
    
    # 5. Process chunks
    total_chunks = len(chunks)
    print(f"Found {total_chunks} chunks in JSON file.")
    
    # To handle incremental updates efficiently, we fetch all existing IDs
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data['ids'])
    print(f"Found {len(existing_ids)} chunks already in the database.")
    
    docs_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    print("Preparing data for ingestion...")
    for chunk in chunks:
        chunk_index = chunk.get("chunk_index")
        chunk_id = f"chunk_{chunk_index}"
        
        # Skip chunks already in the database
        if chunk_id in existing_ids:
            continue
            
        user_msg = chunk.get("user", "")
        asst_msg = chunk.get("assistant", "")
        document = f"User: {user_msg}\nAssistant: {asst_msg}"
        
        metadata = {
            "title": chunk.get("title", "Untitled"),
            "date": chunk.get("date", "Unknown"),
            "chunk_index": chunk_index
        }
        
        docs_to_add.append(document)
        metadatas_to_add.append(metadata)
        ids_to_add.append(chunk_id)
        
    chunks_embedded = 0
    batch_size = 100 # Batching for efficient DB insertion and embedding progress reporting
    
    total_to_embed = len(docs_to_add)
    
    if total_to_embed == 0:
        print("No new chunks to embed. Database is up to date!")
        return
        
    print(f"Starting embedding for {total_to_embed} new chunks...")
    
    for i in range(0, total_to_embed, batch_size):
        batch_docs = docs_to_add[i:i + batch_size]
        batch_metas = metadatas_to_add[i:i + batch_size]
        batch_ids = ids_to_add[i:i + batch_size]
        
        # This will automatically generate embeddings using the embedding_function
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
        chunks_embedded += len(batch_docs)
        if chunks_embedded % 100 == 0 or chunks_embedded == total_to_embed:
            print(f"Progress: {chunks_embedded} / {total_to_embed} chunks embedded.")
            
    print("\n=== Embedding Complete ===")
    print(f"Total new chunks embedded: {chunks_embedded}")
    print(f"Total chunks in database: {len(existing_ids) + chunks_embedded}")

if __name__ == "__main__":
    main()
