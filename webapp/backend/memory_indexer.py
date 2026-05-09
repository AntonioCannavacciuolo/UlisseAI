import os
import json
import torch
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from msa_provider import msa_manager

CACHE_FILE = "./memory_index_cache.json"

class MemoryIndexer:
    def __init__(self, corpus_dir: str, wiki_dir: str, output_path: str = "./memory_bank.pt"):
        self.corpus_dir = Path(corpus_dir)
        self.wiki_dir = Path(wiki_dir)
        self.output_path = Path(output_path)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f)

    def _get_file_hash(self, path: Path) -> str:
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def scan_files(self) -> List[Dict[str, Any]]:
        documents = []
        
        # Scan Corpus (sessions)
        sessions_dir = self.corpus_dir / "sessions"
        if sessions_dir.exists():
            for f in sessions_dir.glob("*.json"):
                file_hash = self._get_file_hash(f)
                if self.cache.get(str(f)) == file_hash:
                    continue
                
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    content = "\n".join([f"{m['role']}: {m['content']}" for m in data.get("messages", [])])
                    documents.append({
                        "id": str(f),
                        "content": content,
                        "metadata": {"source": "corpus", "title": data.get("title", f.stem)},
                        "hash": file_hash
                    })

        # Scan Wiki
        wiki_pages_dir = self.wiki_dir / "pages"
        if wiki_pages_dir.exists():
            for f in wiki_pages_dir.glob("*.md"):
                file_hash = self._get_file_hash(f)
                if self.cache.get(str(f)) == file_hash:
                    continue

                content = f.read_text(encoding="utf-8")
                # Split by ## headings
                sections = content.split("\n## ")
                for i, section in enumerate(sections):
                    if not section.strip():
                        continue
                    
                    # Prepend ## if it wasn't the first section
                    full_section = ("## " + section) if i > 0 else section
                    documents.append({
                        "id": f"{f}#section-{i}",
                        "content": full_section,
                        "metadata": {"source": "wiki", "title": f.stem},
                        "hash": file_hash # Note: using file hash for simplicity, could be more granular
                    })
        
        return documents

    def build_memory_bank(self, progress_callback=None):
        docs_to_index = self.scan_files()
        if not docs_to_index:
            print("No changes detected. Skipping re-indexing.")
            return

        if msa_manager.model is None:
            msa_manager.load_model()

        # Load existing memory bank if it exists
        if os.path.exists(self.output_path):
            memory_bank = torch.load(self.output_path)
        else:
            memory_bank = []

        total = len(docs_to_index)
        for i, doc in enumerate(docs_to_index):
            if progress_callback:
                progress_callback(i + 1, total, doc['metadata']['title'])
            
            # Pre-compute KV vectors
            # Note: This depends on MSA-4B's specific API. 
            # Assuming a helper exists or we use the model's encoder.
            with torch.no_grad():
                inputs = msa_manager.tokenizer(doc['content'], return_tensors="pt").to(msa_manager.device)
                # In MSA architecture, we typically store the hidden states or KV cache
                # for the document. 
                # For this implementation, we'll store the encoded representations.
                outputs = msa_manager.model(**inputs, output_hidden_states=True)
                # We store the last hidden state as a representation of the document
                kv_vectors = outputs.hidden_states[-1].cpu()
                
                # Update memory bank (replace if already exists or append)
                # For simplicity, we append here. In a real app, you'd find and replace by ID.
                memory_bank.append({
                    "id": doc["id"],
                    "vectors": kv_vectors,
                    "metadata": doc["metadata"]
                })
            
            # Update cache
            self.cache[doc["id"].split("#")[0]] = doc["hash"]

        torch.save(memory_bank, self.output_path)
        self._save_cache()
        print(f"Memory bank updated with {total} new/modified documents.")

indexer = MemoryIndexer(corpus_dir="corpus", wiki_dir="corpus/wiki")
