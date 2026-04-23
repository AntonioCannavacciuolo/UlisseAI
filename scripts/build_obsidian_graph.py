import os
import json
import re
from collections import Counter
from dotenv import load_dotenv

import nltk

def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

setup_nltk()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load .env
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(env_key, default_folder):
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        return env_path
    # Fallback to local directory relative to BASE_DIR
    return os.path.join(BASE_DIR, default_folder)

corpus_path = get_path("CORPUS_PATH", "corpus")
vault_path = get_path("VAULT_PATH", "vault")
vectordb_path = get_path("VECTORDB_PATH", "vectordb")

print(f"Using paths:\nCorpus: {corpus_path}\nVault: {vault_path}\nVectorDB: {vectordb_path}")

# Read JSON chunks
text_chunks = []

corpus_clean = os.path.join(vectordb_path, "corpus_clean.json")
if os.path.exists(corpus_clean):
    print(f"Reading {corpus_clean}...")
    with open(corpus_clean, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            def extract_text(obj):
                if isinstance(obj, str):
                    text_chunks.append(obj)
                elif isinstance(obj, list):
                    for item in obj: extract_text(item)
                elif isinstance(obj, dict):
                    for v in obj.values(): extract_text(v)
            extract_text(data)
        except Exception as e:
            print(f"Error reading {corpus_clean}: {e}")

new_convs = os.path.join(corpus_path, "new_conversations.jsonl")
if os.path.exists(new_convs):
    print(f"Reading {new_convs}...")
    with open(new_convs, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if 'user_message' in obj:
                        text_chunks.append(obj['user_message'])
                    if 'assistant_response' in obj:
                        text_chunks.append(obj['assistant_response'])
                except Exception as e:
                    pass

print(f"Extracted {len(text_chunks)} text chunks.")

stop_words = set(stopwords.words('italian')).union(set(stopwords.words('english')))

all_words = []
proper_nouns_counter = Counter()
general_words_counter = Counter()

for chunk in text_chunks:
    words = word_tokenize(chunk)
    # Filter punctuation and keep words > 2 chars
    words = [w for w in words if w.isalpha() and len(w) > 2]
    
    for w in words:
        w_lower = w.lower()
        if w_lower not in stop_words:
            if w[0].isupper():
                proper_nouns_counter[w] += 1
            else:
                general_words_counter[w_lower] += 1

valid_concepts = set()
for w, count in proper_nouns_counter.items():
    valid_concepts.add(w)

for w, count in general_words_counter.items():
    if count >= 3:
        valid_concepts.add(w)

# Sort concepts by length descending
valid_concepts = sorted(list(valid_concepts), key=len, reverse=True)
print(f"Identified {len(valid_concepts)} concepts.")

concepts_dir = os.path.join(vault_path, "concepts")
os.makedirs(concepts_dir, exist_ok=True)

concept_to_convs = {c: set() for c in valid_concepts}
total_conversations = 0

if os.path.exists(vault_path):
    for filename in os.listdir(vault_path):
        if not filename.endswith('.md'): continue
        if filename == 'ULISSE_MAP.md': continue
        
        filepath = os.path.join(vault_path, filename)
        if os.path.isdir(filepath): continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        conv_title = filename[:-3]
        total_conversations += 1
        
        modified = False
        for concept in valid_concepts:
            # We use a pattern that checks for whole words not preceded by [[
            pattern = rf'(?<!\[\[)\b({re.escape(concept)})\b(?!\]\])'
            # Fast check
            if concept.lower() in content.lower():
                matches = list(re.finditer(pattern, content, flags=re.IGNORECASE))
                if matches:
                    content = re.sub(pattern, r'[[\g<1>]]', content, flags=re.IGNORECASE)
                    modified = True
                    # Add to the concept mapping
                    concept_to_convs[concept].add(conv_title)
                
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

active_concepts = {c: convs for c, convs in concept_to_convs.items() if convs}
print(f"Found {len(active_concepts)} active concepts in vault notes.")

total_connections = 0
for concept, convs in active_concepts.items():
    mentions = len(convs)
    total_connections += mentions
    safe_concept = "".join(x for x in concept if x.isalnum() or x in " -_")
    if not safe_concept: continue
    concept_file = os.path.join(concepts_dir, f"{safe_concept}.md")
    
    with open(concept_file, 'w', encoding='utf-8') as f:
        f.write(f"---\n")
        f.write(f"type: concept\n")
        f.write(f"mentions: {mentions}\n")
        f.write(f"---\n")
        f.write(f"# {concept}\n")
        f.write(f"## Appare in:\n")
        for conv in sorted(convs):
            f.write(f"- [[{conv}]]\n")

map_path = os.path.join(vault_path, "ULISSE_MAP.md")
with open(map_path, 'w', encoding='utf-8') as f:
    f.write("# ULISSE KNOWLEDGE MAP\n\n")
    f.write("## Statistiche\n")
    f.write(f"- **Conversazioni:** {total_conversations}\n")
    f.write(f"- **Concetti:** {len(active_concepts)}\n")
    f.write(f"- **Connessioni:** {total_connections}\n\n")
    
    f.write("## Concetti (per frequenza)\n")
    sorted_concepts = sorted(active_concepts.items(), key=lambda x: len(x[1]), reverse=True)
    for concept, convs in sorted_concepts:
        f.write(f"- [[{concept}]] ({len(convs)} menzioni)\n")

print("Obsidian graph built successfully.")
