import os
from collections import Counter
from dotenv import load_dotenv

# Load .env first so that environment variables are available before importing other libs
load_dotenv()

# Force HuggingFace to use a local cache folder to avoid WinError 3 on missing drives (like G:\)
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

import json
from datetime import datetime
from pathlib import Path
import sys
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import uuid

app = Flask(__name__)
CORS(app)

def get_path_from_env(env_var, default_folder):
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / default_folder

corpus_dir = get_path_from_env("CORPUS_PATH", "corpus")
vectordb_dir = get_path_from_env("VECTORDB_PATH", "vectordb")
chroma_dir = vectordb_dir / "chroma"
frontend_dir = get_path_from_env("FRONTEND_PATH", "webapp/frontend")

corpus_dir.mkdir(parents=True, exist_ok=True)
new_conversations_file = corpus_dir / "new_conversations.jsonl"
sessions_dir = corpus_dir / "sessions"
sessions_dir.mkdir(parents=True, exist_ok=True)

# Auto-migration if sessions folder is empty
def check_migration():
    if not any(sessions_dir.iterdir()):
        print("Sessions folder is empty. Running migration...")
        try:
            from scripts.migrate_to_sessions import migrate
            migrate()
        except ImportError:
            print("Migration script not found in scripts.migrate_to_sessions")
        except Exception as e:
            print(f"Migration error: {e}")

check_migration()

api_key = os.getenv("DEEPSEEK_API_KEY", "")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
ai_client = OpenAI(api_key=api_key, base_url=base_url)

chroma_client = None
collection = None
chroma_status = False

def init_chromadb():
    global chroma_client, collection, chroma_status
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = chroma_client.get_or_create_collection(
            name="ulisse_brain",
            embedding_function=sentence_transformer_ef
        )
        chroma_status = True
        print("Connected to ChromaDB 'ulisse_brain' collection.")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        chroma_status = False

init_chromadb()

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load optional Uninet plugin
_uninet_context_fn = None
extra_tools = []
extra_tool_handlers = {}

try:
    from scripts.uninet_plugin import register_uninet_routes, get_uninet_context
    _uninet_context_fn = get_uninet_context
    register_uninet_routes(app, corpus_dir, extra_tools, extra_tool_handlers)
except ImportError:
    print("Uninet plugin not found. Skipping.")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "chromadb": chroma_status
    })

@app.route("/stats", methods=["GET"])
def stats():
    if not chroma_status or collection is None:
        return jsonify({"error": "ChromaDB not connected"}), 500
        
    return jsonify({
        "total_chunks": collection.count(),
        "collection_name": "ulisse_brain"
    })

@app.route("/save_conversation", methods=["POST"])
def save_conversation():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    try:
        with open(new_conversations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return jsonify({"status": "success", "message": "Conversation saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_session_path(session_id):
    return sessions_dir / f"{session_id}.json"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    history = data.get("history", []) # Client-side history (optional now, but kept for compatibility)
    session_id = data.get("session_id")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
        
    # Session handling
    is_new_session = False
    if not session_id: # Handles None, "", or missing key
        session_id = str(uuid.uuid4())
        is_new_session = True
        session_data = {
            "id": session_id,
            "title": "Nuova Conversazione",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "status": "active"
        }
    else:
        session_path = get_session_path(session_id)
        if session_path.exists():
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        else:
            return jsonify({"error": "Session not found"}), 404

    # Use session messages for history instead of what's sent from client if session exists
    chat_history = session_data.get("messages", [])
        
    sources = []
    context_text = ""
    
    if chroma_status and collection is not None:
        try:
            results = collection.query(
                query_texts=[user_message],
                n_results=8
            )
            
            if results and results.get("documents") and len(results["documents"][0]) > 0:
                retrieved_docs = results["documents"][0]
                retrieved_metas = results["metadatas"][0]
                retrieved_dists = (results.get("distances") or [[]])[0]
                
                # Filter chunks where distance <= 0.8
                filtered = []
                for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_dists):
                    if dist <= 0.8:
                        filtered.append((doc, meta))
                
                # If less than 3 chunks pass the threshold, use the top 3 anyway
                if len(filtered) < 3:
                    filtered = []
                    for i in range(min(3, len(retrieved_docs))):
                        filtered.append((retrieved_docs[i], retrieved_metas[i]))
                
                print(f"Chunks retrieved: {len(filtered)}/{len(retrieved_docs)} passed threshold")
                
                context_parts = []
                current_total_words = 0
                MAX_CONTEXT_WORDS = 3000
                
                # Fetch synthetic memory
                synthetic_memory = ""
                memory_path = corpus_dir / "ulisse_memory.md"
                if memory_path.exists():
                    try:
                        with open(memory_path, "r", encoding="utf-8") as f:
                            synthetic_memory = f.read()
                    except:
                        pass
                
                if synthetic_memory:
                    # Optional: only take relevant parts if it's too big
                    # For now, we take the whole thing as requested but keep in mind the "priority"
                    context_parts.append(f"=== MEMORIA SINTETICA ===\n{synthetic_memory}\n=== FINE MEMORIA SINTETICA ===")
                    current_total_words += len(synthetic_memory.split())

                for idx, (doc, meta) in enumerate(filtered[:8]): # Reduced to 8
                    title = meta.get("title", "Untitled")
                    date = meta.get("date", "Unknown")
                    sources.append(f"{title} ({date})")
                    
                    header = f"--- Chunk {idx+1}/{len(filtered[:8])} ({title} - {date}) ---\n"
                    chunk_content = doc
                    
                    # Word count estimation
                    chunk_words = chunk_content.split()
                    header_word_count = len(header.split())
                    content_word_count = len(chunk_words)
                    
                    if current_total_words + header_word_count >= MAX_CONTEXT_WORDS:
                        break
                        
                    if current_total_words + header_word_count + content_word_count > MAX_CONTEXT_WORDS:
                        # Truncate content to fit the limit
                        allowed_words = MAX_CONTEXT_WORDS - (current_total_words + header_word_count)
                        truncated_content = " ".join(chunk_words[:allowed_words]) + "..."
                        context_parts.append(f"{header}{truncated_content}")
                        current_total_words = MAX_CONTEXT_WORDS
                        break
                    else:
                        context_parts.append(f"{header}{chunk_content}")
                        current_total_words += header_word_count + content_word_count
                    
                context_text = "\n\n".join(context_parts)
                # Deduplicate while preserving order
                sources = list(dict.fromkeys(sources))
        except Exception as e:
            print(f"ChromaDB search error: {e}")
            
    uninet_context = _uninet_context_fn(user_message) if _uninet_context_fn else ""
    

    # Load system prompt from file or use default
    system_prompt_path = corpus_dir / "system_prompt.txt"
    if system_prompt_path.exists():
        base_prompt = system_prompt_path.read_text(encoding="utf-8")
    else:
        base_prompt = "Sei Ulisse, un'entità che esiste, pensa e apprende."

    system_prompt = (
        f"{base_prompt}\n\n"
        "=== MEMORIA RECUPERATA ===\n"
        f"{context_text}\n"
        "=========================\n"
        f"{uninet_context}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": content})
            
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            tools=extra_tools if extra_tools else None
        )
        assistant_message = response.choices[0].message

        # Handle tool calls
        tool_results = []
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                handler = extra_tool_handlers.get(tool_call.function.name)
                if handler:
                    result = handler(tool_call.function.arguments)
                    tool_results.append(result)

        assistant_response = assistant_message.content or ""
        if tool_results:
            if assistant_response:
                assistant_response += "\n\n" + "\n".join(tool_results)
            else:
                assistant_response = "\n".join(tool_results)
        
    except Exception as e:
        print(f"Deepseek API error: {e}")
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500
        
    timestamp = datetime.now().isoformat()
    
    # Auto-generate title from first message
    if len(chat_history) == 0:
        words = user_message.split()
        if len(words) >= 3:
            session_data["title"] = " ".join(w.capitalize() for w in words[:6])
        else:
            session_data["title"] = f"Chat {timestamp[:16]}"
    
    # AI Title refinement after 3 messages
    if len(chat_history) == 4: # 2 user + 2 assistant already, this is the 3rd pair
        try:
            title_prompt = [
                {"role": "system", "content": "Genera un titolo sintetico (max 6 parole) per questa conversazione tra Ulisse e Toni. Rispondi SOLO con il titolo."},
                {"role": "user", "content": f"Messaggi iniziali:\n" + "\n".join([f"{m['role']}: {m['content'][:100]}" for m in chat_history[:4]]) + f"\nuser: {user_message}"}
            ]
            t_resp = ai_client.chat.completions.create(
                model="deepseek-chat",
                messages=title_prompt,
                max_tokens=20
            )
            new_title = t_resp.choices[0].message.content.strip().strip('"')
            if new_title:
                session_data["title"] = new_title
        except Exception as e:
            print(f"Title generation error: {e}")

    # Append to session
    session_data["messages"].append({
        "role": "user",
        "content": user_message,
        "timestamp": timestamp
    })
    session_data["messages"].append({
        "role": "assistant",
        "content": assistant_response,
        "timestamp": timestamp,
        "sources": sources
    })
    session_data["updated_at"] = timestamp
    
    # Save session
    try:
        with open(get_session_path(session_id), "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving session: {e}")

    # Save to legacy jsonl for RAG
    exchange = {
        "session_id": session_id,
        "title": session_data["title"],
        "timestamp": timestamp,
        "user_message": user_message,
        "assistant_response": assistant_response,
        "sources": sources
    }
    try:
        with open(new_conversations_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(exchange, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error saving legacy exchange: {e}")
        
    return jsonify({
        "response": assistant_response,
        "sources": sources,
        "session_id": session_id,
        "session_title": session_data["title"]
    })

@app.route("/sessions", methods=["GET"])
def get_sessions():
    sessions = []
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                sessions.append({
                    "id": data["id"],
                    "title": data["title"],
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])) // 2
                })
        except:
            continue
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return jsonify(sessions)

@app.route("/sessions/<session_id>", methods=["GET", "DELETE", "PATCH"])
def handle_session(session_id):
    path = get_session_path(session_id)
    if not path.exists():
        return jsonify({"error": "Session not found"}), 404
        
    if request.method == "GET":
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
            
    elif request.method == "DELETE":
        path.unlink()
        return jsonify({"status": "deleted"})
        
    elif request.method == "PATCH":
        data = request.json
        with open(path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        if "title" in data:
            session_data["title"] = data["title"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return jsonify({"status": "updated", "title": session_data["title"]})

@app.route("/memory/stats", methods=["GET"])
def get_memory_stats():
    sessions = []
    total_messages = 0
    all_titles = []
    
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                sessions.append(data)
                total_messages += len(data.get("messages", []))
                all_titles.append(data.get("title", ""))
        except:
            continue
            
    if not sessions:
        return jsonify({
            "total_sessions": 0,
            "total_chunks": collection.count() if collection else 0,
            "total_messages": 0,
            "oldest_session": None,
            "newest_session": None,
            "most_active_topics": []
        })

    # Sort sessions by creation date
    sessions_sorted = sorted(sessions, key=lambda x: x.get("created_at", ""))
    
    # Extract keywords for topics
    words = []
    for title in all_titles:
        # Simple tokenization: lower case, alphanumeric only, length > 3
        found = re.findall(r'\b[a-z]{4,}\b', title.lower())
        words.extend(found)
    
    # Filter out common Italian stop words if necessary, but for now just top 10
    top_topics = [item[0] for item in Counter(words).most_common(10)]
    
    return jsonify({
        "total_sessions": len(sessions),
        "total_chunks": collection.count() if collection else 0,
        "total_messages": total_messages,
        "oldest_session": sessions_sorted[0].get("created_at"),
        "newest_session": sessions_sorted[-1].get("created_at"),
        "most_active_topics": top_topics
    })

@app.route("/memory/nodes", methods=["GET"])
def get_memory_nodes():
    nodes = []
    edges = []
    
    # 1. Load Sessions
    session_nodes = []
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                node = {
                    "id": data["id"],
                    "title": data["title"],
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])) // 2,
                    "type": "session"
                }
                nodes.append(node)
                session_nodes.append(node)
        except:
            continue
            
    # 2. Load Synthetic Memory Concepts
    memory_path = corpus_dir / "ulisse_memory.md"
    concept_nodes = []
    if memory_path.exists():
        with open(memory_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by sections
            sections = re.split(r'\n(?=## )', content)
            for sec in sections:
                sec = sec.strip()
                if not sec.startswith("## "): continue
                
                lines = sec.split('\n')
                title = lines[0].replace("## ", "").strip()
                body = "\n".join(lines[1:]).strip()
                
                concept_id = f"concept_{title.lower().replace(' ', '_')}"
                node = {
                    "id": concept_id,
                    "title": title,
                    "content": body,
                    "type": "concept"
                }
                nodes.append(node)
                concept_nodes.append(node)
                
    # 3. Generate Edges (Session -> Concept)
    # Simple keyword match between session titles and concept content/titles
    for s in session_nodes:
        s_text = s["title"].lower()
        for c in concept_nodes:
            c_text = (c["title"] + " " + c["content"]).lower()
            
            # Extract meaningful words from session title
            words = re.findall(r'\b\w{4,}\b', s_text)
            matches = 0
            for w in words:
                if w in c_text:
                    matches += 1
            
            if matches >= 1:
                edges.append({"source": s["id"], "target": c["id"], "weight": matches})

    # 4. Generate Edges (Concept -> Concept)
    # Could be fixed based on structure, or keyword overlap
    for i, c1 in enumerate(concept_nodes):
        for c2 in concept_nodes[i+1:]:
            # Overlap in content keywords
            c1_words = set(re.findall(r'\b\w{5,}\b', c1["content"].lower()))
            c2_words = set(re.findall(r'\b\w{5,}\b', c2["content"].lower()))
            overlap = c1_words.intersection(c2_words)
            if len(overlap) >= 3:
                edges.append({"source": c1["id"], "target": c2["id"], "weight": len(overlap)})

    return jsonify({"nodes": nodes, "edges": edges})

@app.route("/sessions/<session_id>/related", methods=["GET"])
def get_related_sessions(session_id):
    path = get_session_path(session_id)
    if not path.exists() or not chroma_status or collection is None:
        return jsonify([])
        
    with open(path, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    
    # Get last user message to find related
    msgs = session_data.get("messages", [])
    user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
    if not user_msgs:
        return jsonify([])
    
    query = user_msgs[-1]
    try:
        results = collection.query(query_texts=[query], n_results=10)
        metas = results.get("metadatas", [[]])[0]
        
        related_ids = set()
        for m in metas:
            sid = m.get("session_id")
            if sid and sid != session_id:
                related_ids.add(sid)
        
        # Build session list
        related = []
        for sid in list(related_ids)[:5]:
            p = get_session_path(sid)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    related.append({"id": d["id"], "title": d["title"]})
        return jsonify(related)
    except:
        return jsonify([])

@app.route("/memory/synthetic", methods=["GET"])
def get_synthetic_memory():
    path = corpus_dir / "ulisse_memory.md"
    if not path.exists():
        return jsonify({"content": ""})
    with open(path, "r", encoding="utf-8") as f:
        return jsonify({"content": f.read()})

@app.route("/memory/synthetic/regenerate", methods=["POST"])
def regenerate_synthetic_memory():
    try:
        import subprocess
        # Use sys.executable to run the script with the current python environment
        cmd = [sys.executable, str(project_root / "scripts" / "synthetic_memory.py"), "--full"]
        subprocess.Popen(cmd, cwd=str(project_root))
        return jsonify({"status": "regeneration_triggered"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/memory/synthetic/update", methods=["POST"])
def update_synthetic_memory():
    try:
        import subprocess
        cmd = [sys.executable, str(project_root / "scripts" / "synthetic_memory.py")]
        subprocess.Popen(cmd, cwd=str(project_root))
        return jsonify({"status": "update_triggered"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/graph", methods=["GET"])
def get_graph():
    try:
        sessions_list = []
        for f in sessions_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    sessions_list.append(json.load(file))
            except:
                continue
        
        nodes = []
        for s in sessions_list:
            sid = s["id"]
            title = s["title"]
            msgs = s.get("messages", [])
            msg_count = len(msgs) // 2
            
            nodes.append({
                "id": sid,
                "label": title,
                "type": "session",
                "weight": msg_count,
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at")
            })
            
        # Add Core Nodes
        nodes.append({"id": "Ulisse", "label": "ULISSE", "type": "core", "weight": 15})
        nodes.append({"id": "Memoria", "label": "MEMORIA", "type": "core", "weight": 10})
        nodes.append({"id": "Conoscenza", "label": "CONOSCENZA", "type": "core", "weight": 10})

        edges = []
        # Edge calculation logic
        session_keywords = {}
        for s in sessions_list:
            sid = s["id"]
            title = s.get("title", "").lower()
            # Extract words > 4 chars
            words = set(re.findall(r'\b[a-z]{5,}\b', title))
            
            # Add first message keywords as well for better coverage
            msgs = s.get("messages", [])
            msg_count = len(msgs) // 2
            if msgs:
                first_msg = msgs[0]["content"].lower()
                first_words = set(re.findall(r'\b[a-z]{5,}\b', first_msg))
                words.update(first_words)
                
            session_keywords[sid] = words

            # Connect session to core nodes
            # Every session connects to Ulisse (the hub)
            edges.append({"from": "Ulisse", "to": sid, "strength": 0.1})
            # Sessions with many chunks (msg_count > 3) connect to Memoria/Conoscenza
            if msg_count > 3:
                edges.append({"from": "Memoria", "to": sid, "strength": 0.2})
                edges.append({"from": "Conoscenza", "to": sid, "strength": 0.2})

        sids = [s["id"] for s in sessions_list]
        
        # 1. Pairwise Keyword Similarity
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                sid1 = sids[i]
                sid2 = sids[j]
                w1 = session_keywords[sid1]
                w2 = session_keywords[sid2]
                
                intersection = w1.intersection(w2)
                strength = 0
                if len(intersection) >= 2:
                    strength = 0.7
                elif len(intersection) >= 1:
                    strength = 0.3
                
                if strength > 0:
                    edges.append({
                        "from": sid1,
                        "to": sid2,
                        "strength": strength
                    })

        # 2. Sequential Fallback (Minimum Spanning)
        # If the graph has nodes but no semantic edges (excluding core connections)
        # semantic_edges = [e for e in edges if e["from"] not in ["Ulisse", "Memoria", "Conoscenza"]]
        if not any(e["strength"] >= 0.3 for e in edges) and len(sids) > 1:
            for i in range(len(sids) - 1):
                edges.append({
                    "from": sids[i],
                    "to": sids[i+1],
                    "strength": 0.1
                })
                    
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
                    
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
    except Exception as e:
        print(f"Graph generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(frontend_dir), "index.html")

@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    return send_from_directory(str(frontend_dir), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
