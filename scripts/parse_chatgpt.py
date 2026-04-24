import os
import json
import re
import glob
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

def sanitize_filename(name):
    """Sanitize the title to be a valid filesystem filename."""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

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
    corpus_dir = get_path_from_env("CORPUS_PATH", "corpus")
    vault_dir = get_path_from_env("VAULT_PATH", "vault")
    vectordb_dir = get_path_from_env("VECTORDB_PATH", "vectordb")
    
    # Ensure output directories exist
    vault_dir.mkdir(parents=True, exist_ok=True)
    vectordb_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    file_patterns = ["conversations.json", "conversations-*.json"]
    files_to_process = []
    for pattern in file_patterns:
        files_to_process.extend(glob.glob(str(corpus_dir / pattern)))
    
    # Sort files to ensure deterministic processing
    files_to_process = sorted(list(set(files_to_process)))
    
    if not files_to_process:
        print(f"No ChatGPT export files found in: {corpus_dir}")
        print("Please place 'conversations.json' or 'conversations-*.json' files in the corpus directory.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    total_conv_processed = 0
    all_chunks = []
    total_vault_notes = 0
    
    for conversations_file in files_to_process:
        print(f"Processing: {os.path.basename(conversations_file)}...")
        # Read the JSON file
        with open(conversations_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing {os.path.basename(conversations_file)}. Skipping.")
                continue
                
        for conv in data:
            title = conv.get("title") or "Untitled"
            create_time = conv.get("create_time") or 0
            date_str = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else "Unknown"
            
            messages = []
            mapping = conv.get("mapping", {})
            
            # Extract messages
            for node_id, node in mapping.items():
                msg = node.get("message")
                if msg and isinstance(msg, dict):
                    author = msg.get("author", {})
                    role = author.get("role")
                    content = msg.get("content", {})
                    
                    if role in ("user", "assistant"):
                        parts = content.get("parts", [])
                        if isinstance(parts, list):
                            # Clean text (remove nulls, strip whitespace)
                            text = "".join([str(p) for p in parts if p is not None]).strip()
                            if text:
                                msg_time = msg.get("create_time") or 0
                                messages.append({
                                    "role": role,
                                    "content": text,
                                    "time": msg_time
                                })
                                
            # Skip empty or very short conversations (less than 2 messages)
            if len(messages) < 2:
                continue
                
            # Sort messages by time to reconstruct chronological flow
            messages.sort(key=lambda x: x["time"])
            
            # --- Save to Markdown in Vault ---
            safe_title = sanitize_filename(title)
            if not safe_title:
                safe_title = f"Conversation_{create_time}"
                
            # Handle filename collisions
            md_filename = f"{safe_title}.md"
            md_path = vault_dir / md_filename
            counter = 1
            while md_path.exists():
                md_filename = f"{safe_title}_{counter}.md"
                md_path = vault_dir / md_filename
                counter += 1
                
            # Build Markdown content
            md_content = []
            md_content.append("---")
            md_content.append(f"title: {title}")
            md_content.append(f"date: {date_str}")
            md_content.append("tags: [chatgpt, imported]")
            md_content.append("---")
            md_content.append(f"# {title}")
            md_content.append("")
            md_content.append("## Messages")
            md_content.append("")
            
            current_user_msg = None
            
            for msg in messages:
                role_capitalized = "User" if msg["role"] == "user" else "Assistant"
                md_content.append(f"**{role_capitalized}:** {msg['content']}")
                md_content.append("")
                
                # --- Build chunks for VectorDB ---
                if msg["role"] == "user":
                    current_user_msg = msg["content"]
                elif msg["role"] == "assistant" and current_user_msg:
                    # We found a user-assistant pair
                    all_chunks.append({
                        "title": title,
                        "date": date_str,
                        "chunk_index": len(all_chunks) + 1,
                        "user": current_user_msg,
                        "assistant": msg["content"]
                    })
                    # Reset to ensure we pair only consecutive or closest pairs
                    current_user_msg = None
                    
            # Write Markdown file
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_content))
                
            total_conv_processed += 1
            total_vault_notes += 1
        
    # --- Save Chunks to JSON ---
    chunks_path = vectordb_dir / "corpus_clean.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    # --- Print Summary ---
    print("\n=== Parsing Complete ===")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Total conversations processed: {total_conv_processed}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Total vault notes created: {total_vault_notes}")

if __name__ == "__main__":
    main()
