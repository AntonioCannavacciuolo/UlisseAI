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
wiki_dir = corpus_dir / "wiki"
wiki_pages_dir = wiki_dir / "pages"
wiki_pages_dir.mkdir(parents=True, exist_ok=True)
wiki_raw_dir = wiki_dir / "raw"
wiki_raw_dir.mkdir(parents=True, exist_ok=True)


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

# --- Wiki Tools Implementation ---
def wiki_read_page(args):
    try:
        data = json.loads(args)
        title = data.get("title", "").strip()
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        path = wiki_pages_dir / f"{safe_title}.md"
        if path.exists():
            return f"Contenuto di [[{title}]]:\n\n" + path.read_text(encoding="utf-8")
        return f"La pagina [[{title}]] non esiste ancora."
    except Exception as e:
        return f"Errore lettura wiki: {str(e)}"

def wiki_write_page(args):
    try:
        data = json.loads(args)
        title = data.get("title", "").strip()
        content = data.get("content", "").strip()
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        path = wiki_pages_dir / f"{safe_title}.md"
        path.write_text(content, encoding="utf-8")
        return f"Pagina [[{title}]] aggiornata con successo."
    except Exception as e:
        return f"Errore scrittura wiki: {str(e)}"

def wiki_list_pages(args=None):
    try:
        pages = [f.stem.replace('_', ' ') for f in wiki_pages_dir.glob("*.md")]
        return "Pagine presenti nella Wiki:\n" + "\n".join([f"- [[{p}]]" for p in pages])
    except Exception as e:
        return f"Errore lista wiki: {str(e)}"

def wiki_append_log(args):
    try:
        data = json.loads(args)
        entry = data.get("entry", "").strip()
        log_path = wiki_dir / "log.md"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n- [{datetime.now().strftime('%Y-%m-%d %H:%M')}] {entry}")
        return "Log aggiornato."
    except Exception as e:
        return f"Errore log wiki: {str(e)}"

def wiki_update_index(args):
    try:
        data = json.loads(args)
        content = data.get("content", "").strip()
        index_path = wiki_dir / "index.md"
        index_path.write_text(content, encoding="utf-8")
        return "Indice Wiki aggiornato."
    except Exception as e:
        return f"Errore indice wiki: {str(e)}"

wiki_tools = [
    {
        "type": "function",
        "function": {
            "name": "wiki_read_page",
            "description": "Legge il contenuto di una pagina della memoria a lungo termine (Wiki).",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Il titolo della pagina da leggere."}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_write_page",
            "description": "Crea o aggiorna una pagina nella memoria a lungo termine (Wiki). Usala per memorizzare informazioni importanti, progetti, fatti o sintesi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Il titolo della pagina."},
                    "content": {"type": "string", "description": "Il contenuto in formato markdown."}
                },
                "required": ["title", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_list_pages",
            "description": "Elenca tutte le pagine presenti nella memoria a lungo termine.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_append_log",
            "description": "Aggiunge un'entrata al registro delle attività (log.md) della Wiki.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string", "description": "Descrizione dell'attività svolta."}
                },
                "required": ["entry"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_update_index",
            "description": "Aggiorna l'indice centrale (index.md) della Wiki.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Il nuovo contenuto completo dell'indice."}
                },
                "required": ["content"]
            }
        }
    }
]

extra_tools.extend(wiki_tools)
extra_tool_handlers.update({
    "wiki_read_page": wiki_read_page,
    "wiki_write_page": wiki_write_page,
    "wiki_list_pages": wiki_list_pages,
    "wiki_append_log": wiki_append_log,
    "wiki_update_index": wiki_update_index
})


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
                
                # Short-term memory is now handled solely via ChromaDB RAG


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

    # Load Wiki Schema for guidance
    wiki_schema = ""
    schema_path = wiki_dir / "WIKI_SCHEMA.md"
    if schema_path.exists():
        wiki_schema = schema_path.read_text(encoding="utf-8")

    system_prompt = (
        f"{base_prompt}\n\n"
        "=== MISSIONE: MANUTENTORE WIKI ===\n"
        "Sei il custode della memoria a lungo termine di Ulisse. "
        "Devi decidere autonomamente (o su richiesta) quando memorizzare informazioni importanti nella Wiki. "
        "Segui rigorosamente lo schema fornito sotto per la gestione della Wiki.\n\n"
        f"{wiki_schema}\n\n"
        "=== MEMORIA RECUPERATA (STM) ===\n"
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
        # Initial call
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            tools=extra_tools if extra_tools else None
        )
        assistant_message = response.choices[0].message
        
        # Recursive tool call handling
        while assistant_message.tool_calls:
            # Add assistant's tool call message to history
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                handler = extra_tool_handlers.get(tool_call.function.name)
                if handler:
                    result = handler(tool_call.function.arguments)
                    # Add tool result message to history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })
            
            # Call again with tool results
            response = ai_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=extra_tools if extra_tools else None
            )
            assistant_message = response.choices[0].message

        assistant_response = assistant_message.content or "Operazione wiki completata."
        
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
            
        # Real-time embedding into ChromaDB (Short-term memory)
        if chroma_status and collection is not None:
            document = f"User: {user_message}\nAssistant: {assistant_response}"
            chunk_id = f"chunk_{session_id}_{len(session_data['messages']) // 2}"
            collection.add(
                documents=[document],
                metadatas=[{
                    "session_id": session_id,
                    "title": session_data["title"],
                    "date": timestamp,
                    "type": "conversation"
                }],
                ids=[chunk_id]
            )
            print(f"Real-time STM update: Added chunk {chunk_id} to ChromaDB.")
            
    except Exception as e:
        print(f"Error saving legacy exchange or updating STM: {e}")

        
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
    
    # Load Wiki Pages instead of sessions for the Memory View
    wiki_pages = list(wiki_pages_dir.glob("*.md"))
    
    for f in wiki_pages:
        try:
            title = f.stem.replace('_', ' ')
            content = f.read_text(encoding="utf-8")
            
            # Extract basic stats
            word_count = len(content.split())
            
            node = {
                "id": f.stem,
                "title": title,
                "content": content,
                "type": "wiki",
                "weight": max(5, min(word_count // 10, 20)),
                "created_at": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
                "updated_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            nodes.append(node)
        except:
            continue
            
    # Generate Edges based on Wiki Links [[Page Name]]
    page_ids = [n["id"] for n in nodes]
    for n in nodes:
        content = n["content"]
        # Find all [[...]] links
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        for link in links:
            target_id = re.sub(r'[^\w\s-]', '', link).strip().replace(' ', '_')
            if target_id in page_ids and target_id != n["id"]:
                edges.append({
                    "source": n["id"],
                    "target": target_id,
                    "weight": 1
                })
                
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

# Synthetic memory routes removed as per user request


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
                "weight": max(5, min(msg_count * 2, 20)),
                "created_at": s.get("created_at"),
                "updated_at": s.get("updated_at")
            })
            
        edges = []
        if chroma_status and collection is not None and len(sessions_list) > 1:
            # Generate edges based on RAG similarity
            for i, s1 in enumerate(sessions_list):
                sid1 = s1["id"]
                title1 = s1.get("title", "")
                
                try:
                    # Query ChromaDB for sessions similar to this one's title or first message
                    query_text = title1
                    if s1.get("messages"):
                        query_text += " " + s1["messages"][0]["content"]
                    
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=5,
                        where={"session_id": {"$ne": sid1}}
                    )
                    
                    if results and results.get("metadatas"):
                        metas = results["metadatas"][0]
                        dists = results.get("distances", [[]])[0]
                        
                        for meta, dist in zip(metas, dists):
                            sid2 = meta.get("session_id")
                            if sid2:
                                # Connection strength based on distance (closer to 0 is stronger)
                                strength = max(0.1, 1.0 - dist)
                                edges.append({
                                    "from": sid1,
                                    "to": sid2,
                                    "strength": strength
                                })
                except Exception as e:
                    print(f"Error calculating graph edges for {sid1}: {e}")
        
        # Deduplicate edges (A->B and B->A might exist)
        unique_edges = {}
        for e in edges:
            pair = tuple(sorted([e["from"], e["to"]]))
            if pair not in unique_edges or e["strength"] > unique_edges[pair]["strength"]:
                unique_edges[pair] = e
        
        return jsonify({
            "nodes": nodes,
            "edges": list(unique_edges.values())
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
