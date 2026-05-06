import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Set HF_HOME explicitly for huggingface models
load_dotenv()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

import chromadb
from chromadb.utils import embedding_functions

def get_path_from_env(env_var, default_folder):
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / default_folder

def main():
    corpus_dir = get_path_from_env("CORPUS_PATH", "corpus")
    vectordb_dir = get_path_from_env("VECTORDB_PATH", "vectordb")
    chroma_dir = vectordb_dir / "chroma"
    
    new_conversations_file = corpus_dir / "new_conversations.jsonl"
    sync_state_file = vectordb_dir / "sync_state.json"
    
    if not new_conversations_file.exists():
        print("No new_conversations.jsonl found. Nothing to sync.")
        return
        
    last_line = 0
    if sync_state_file.exists():
        try:
            with open(sync_state_file, "r") as f:
                state = json.load(f)
                last_line = state.get("last_line_processed", 0)
        except Exception as e:
            print(f"Could not read sync state, starting from 0. Error: {e}")
            last_line = 0
            
    with open(new_conversations_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = lines[last_line:]
    if not new_lines:
        print("No new lines to process.")
        return
        
    print(f"Connecting to ChromaDB at {chroma_dir}...")
    try:
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.get_or_create_collection(
            name="ulisse_brain",
            embedding_function=sentence_transformer_ef
        )
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return
        
    current_count = collection.count()
    
    docs_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    lines_processed_this_run = 0
    
    for idx, line in enumerate(new_lines):
        if not line.strip():
            lines_processed_this_run += 1
            continue
            
        try:
            data = json.loads(line)
            user_msg = data.get("user_message", "")
            asst_msg = data.get("assistant_response", "")
            timestamp = data.get("timestamp", "Unknown")
            
            document = f"User: {user_msg}\nAssistant: {asst_msg}"
            
            # Create a unique incremental chunk index
            chunk_idx = current_count + lines_processed_this_run + 1
            
            title = data.get("title")
            if not title:
                words = user_msg.split()
                if len(words) < 3:
                    title = timestamp
                else:
                    title = " ".join(w.capitalize() for w in words[:6])

            session_id = data.get("session_id")
            
            metadata = {
                "title": str(title) if title else "Untitled",
                "date": str(timestamp) if timestamp else "Unknown",
                "chunk_index": int(chunk_idx)
            }
            if session_id:
                metadata["session_id"] = str(session_id)
            
            # Distinctive ID prefix
            chunk_id = data.get("chunk_id") or f"chunk_ulisse_{chunk_idx}"
            
            docs_to_add.append(document)
            metadatas_to_add.append(metadata)
            ids_to_add.append(chunk_id)
            
            lines_processed_this_run += 1
            
        except json.JSONDecodeError:
            print(f"Error parsing line {last_line + idx}: {line}")
            lines_processed_this_run += 1 
            continue

    if docs_to_add:
        print(f"Embedding and adding {len(docs_to_add)} new chunks to ChromaDB...")
        try:
            collection.upsert(
                documents=docs_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print("Added successfully.")
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
            return
    else:
        print("No valid chunks found to add.")
        
    # Commit the new line position to state
    new_state = {
        "last_line_processed": last_line + lines_processed_this_run
    }
    with open(sync_state_file, "w") as f:
        json.dump(new_state, f)
        
    print(f"Sync complete. {len(docs_to_add)} new chunks added.")

if __name__ == "__main__":
    main()
