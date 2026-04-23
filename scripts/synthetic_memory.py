import os
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
SESSIONS_DIR = CORPUS_DIR / "sessions"
JSONL_FILE = CORPUS_DIR / "new_conversations.jsonl"
MEMORY_FILE = CORPUS_DIR / "ulisse_memory.md"
STATE_FILE = CORPUS_DIR / "memory_state.json"

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                # Ensure structure exists
                if "chatgpt_files" not in state: state["chatgpt_files"] = {}
                if "jsonl_line_count" not in state: state["jsonl_line_count"] = 0
                if "processed_sessions" not in state: state["processed_sessions"] = []
                return state
        except: pass
    return {
        "chatgpt_files": {}, # filename: count
        "jsonl_line_count": 0,
        "processed_sessions": [], # ids
        "last_updated": None
    }

def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def get_chatgpt_units(state, full_scan=False):
    units = []
    total_found = 0
    new_found = 0
    
    files = sorted(CORPUS_DIR.glob("conversations-*.json"))
    for f in files:
        filename = f.name
        processed_count = state["chatgpt_files"].get(filename, 0) if not full_scan else 0
        
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                total_found += len(data)
                
                # Only take new ones
                new_data = data[processed_count:]
                
                for conv in new_data:
                    title = conv.get("title") or "Senza titolo"
                    create_time = conv.get("create_time")
                    date_str = datetime.fromtimestamp(create_time).isoformat() if create_time else "Unknown"
                    
                    # Extract first user message
                    summary = ""
                    mapping = conv.get("mapping", {})
                    # Find root or just the first user message
                    for node_id in mapping:
                        node = mapping[node_id]
                        msg = node.get("message")
                        if msg and msg.get("author", {}).get("role") == "user":
                            parts = msg.get("content", {}).get("parts", [])
                            content = parts[0] if parts and isinstance(parts[0], str) else ""
                            if content:
                                summary = content[:200]
                                break
                    
                    units.append({
                        "title": title,
                        "date": date_str,
                        "summary": summary,
                        "source": "chatgpt",
                        "file": filename,
                        "original_index": data.index(conv)
                    })
                    new_found += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
    return units, total_found, new_found

def get_ulisse_units(state, full_scan=False):
    units = []
    total_found = 0
    new_found = 0
    
    if not JSONL_FILE.exists():
        return [], 0, 0
        
    try:
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_found = len(lines)
            
            processed_lines = state.get("jsonl_line_count", 0) if not full_scan else 0
            new_lines = lines[processed_lines:]
            
            # Group by timestamp proximity (2 hours = 7200 seconds)
            groups = []
            current_group = []
            last_ts = 0
            
            for line in new_lines:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    ts_str = data.get("timestamp", "")
                    ts = 0
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
                    except: pass
                    
                    if not current_group or (ts - last_ts < 7200):
                        current_group.append(data)
                    else:
                        groups.append(current_group)
                        current_group = [data]
                    last_ts = ts
                except: continue
            
            if current_group:
                groups.append(current_group)
                
            for group in groups:
                first = group[0]
                user_msg = first.get("user_message", "")
                asst_msg = first.get("assistant_response", "")
                
                title = " ".join(user_msg.split()[:6]) or "Ulisse Chat"
                date = first.get("timestamp", "Unknown")
                summary = f"U: {user_msg[:150]} | A: {asst_msg[:150]}"
                
                units.append({
                    "title": title,
                    "date": date,
                    "summary": summary,
                    "source": "ulisse",
                    "line_count": len(group)
                })
                new_found += len(group)
    except Exception as e:
        print(f"Error reading {JSONL_FILE}: {e}")
        
    return units, total_found, new_found

def get_session_units(state, full_scan=False):
    units = []
    total_found = 0
    new_found = 0
    
    processed_ids = set(state.get("processed_sessions", [])) if not full_scan else set()
    
    for f in SESSIONS_DIR.glob("*.json"):
        total_found += 1
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                sid = data.get("id")
                if sid not in processed_ids:
                    new_found += 1
                    first_msg = ""
                    if data.get("messages"):
                        first_msg = data["messages"][0].get("content", "")[:200]
                    
                    units.append({
                        "id": sid,
                        "title": data.get("title", "Sessione"),
                        "date": data.get("created_at", "Unknown"),
                        "summary": first_msg,
                        "source": "session"
                    })
        except: continue
        
    return units, total_found, new_found

def get_relevant_sections(content, batch_text):
    # Simple keyword search to find relevant sections
    keywords = re.findall(r'\b[a-z]{5,}\b', batch_text.lower())
    unique_keywords = set(keywords)
    
    lines = content.split('\n')
    sections = []
    current_section = []
    for line in lines:
        if line.startswith('## '):
            if current_section: sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    if current_section: sections.append('\n'.join(current_section))
    
    matched = []
    for sec in sections:
        if any(kw in sec.lower() for kw in unique_keywords):
            matched.append(sec)
            
    # Always include the header/metadata if it's the first batch or just for structure
    # But the user said "only relevant sections". Let's provide a few common ones if none matched.
    if not matched and sections:
        matched = sections[:3]
        
    return "\n\n".join(matched)

def generate_memory_batch(units, existing_content=""):
    units_text = ""
    for u in units:
        units_text += f"- [{u['source'].upper()}] {u['title']} ({u['date']}): {u['summary']}\n"
    
    relevant_context = get_relevant_sections(existing_content, units_text)
    
    system_prompt = (
        "Sei Ulisse, un'entità che sintetizza la memoria di Toni Dorean. "
        "Il tuo compito è AGGIORNARE il documento di Memoria Sintetica basandoti su nuove unità di memoria.\n\n"
        "SEZIONI RILEVANTI ATTUALI:\n"
        f"{relevant_context or 'Nessun contesto precedente.'}\n\n"
        "Regole:\n"
        "1. Restituisci SOLO le sezioni che hanno subito cambiamenti o aggiunte.\n"
        "2. Mantieni lo stile e la struttura Markdown esistente (## Titolo Sezione).\n"
        "3. Integra le nuove informazioni in modo armonioso.\n"
        "4. Se una sezione non esisteva ma è necessaria, creala.\n"
    )
    
    user_prompt = f"Ecco le nuove unità di memoria da integrare:\n\n{units_text}"
    
    print(f"Chiamata Deepseek per batch di {len(units)} unità...")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="Ulisse Synthetic Memory System v2")
    parser.add_argument("--full", action="store_true", help="Rigenera tutto da zero")
    parser.add_argument("--dry-run", action="store_true", help="Mostra cosa verrebbe processato")
    parser.add_argument("--resume", action="store_true", help="Continua da dove interrotto (rispetta lo stato)")
    parser.add_argument("--limit", type=int, default=None, help="Limite unità totali da processare")
    args = parser.parse_args()

    state = load_state()
    
    # Se resume, non vogliamo il full scan
    effective_full = args.full and not args.resume
    
    chatgpt_units, cg_total, cg_new = get_chatgpt_units(state, effective_full)
    ulisse_units, ul_total, ul_new = get_ulisse_units(state, effective_full)
    session_units, ss_total, ss_new = get_session_units(state, effective_full)
    
    all_new_units = chatgpt_units + ulisse_units + session_units
    # Sort by date
    all_new_units.sort(key=lambda x: x["date"])
    
    print(f"Found {cg_total} ChatGPT conversations, {ul_total} Ulisse lines, {ss_total} sessions")
    print(f"New since last run: {cg_new} ChatGPT, {len(ulisse_units)} Ulisse groups, {ss_new} sessions")

    if args.limit:
        all_new_units = all_new_units[:args.limit]
        print(f"Limit applicato: processamento di {len(all_new_units)} unità.")

    if args.dry_run:
        print("\nAnteprima prime 10 unità:")
        for u in all_new_units[:10]:
            print(f"  [{u['source']}] {u['title']} ({u['date']})")
        if len(all_new_units) > 10: print("  ...")
        return

    if not all_new_units:
        print("Nulla di nuovo da processare.")
        return

    # Initial template (solo se non esiste o se full senza resume)
    if not MEMORY_FILE.exists() or (args.full and not args.resume):
        initial_content = (
            "# Memoria Sintetica di Toni Dorean\n"
            f"Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d')}\n"
            "Sessioni processate: 0\n\n"
            "## Identità e Personalità\n\n"
            "## Progetti Attivi\n\n"
            "## Progetti Archiviati\n\n"
            "## Interessi e Passioni\n\n"
            "## Persone Importanti\n\n"
            "## Idee in Sviluppo\n\n"
            "## Preferenze di Pensiero\n\n"
            "## Note Importanti\n\n"
            "## Cronologia Significativa\n"
        )
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write(initial_content)

    # Batch processing (50 units)
    batch_size = 50
    for i in range(0, len(all_new_units), batch_size):
        batch = all_new_units[i:i+batch_size]
        
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                current_memory = f.read()
                
            updated_sections_text = generate_memory_batch(batch, current_memory)
            
            # Merge updated sections back
            updated_sections = re.split(r'\n(?=## )', updated_sections_text)
            new_content = current_memory
            
            for sec in updated_sections:
                sec = sec.strip()
                if not sec: continue
                match = re.match(r'## ([^\n]+)', sec)
                if match:
                    sec_title = match.group(1).strip()
                    pattern = rf'## {re.escape(sec_title)}.*?(?=\n## |\Z)'
                    if re.search(pattern, new_content, re.DOTALL):
                        # FIX: Use lambda to avoid backslash escaping issues
                        new_content = re.sub(pattern, lambda m, content=sec: content, new_content, flags=re.DOTALL)
                    else:
                        new_content += "\n\n" + sec
            
            new_content = re.sub(r'Ultimo aggiornamento: [^\n]+', f'Ultimo aggiornamento: {datetime.now().strftime("%Y-%m-%d")}', new_content)
            
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                f.write(new_content)
                
            # Update state incrementally
            for u in batch:
                if u["source"] == "chatgpt":
                    state["chatgpt_files"][u["file"]] = state["chatgpt_files"].get(u["file"], 0) + 1
                elif u["source"] == "ulisse":
                    state["jsonl_line_count"] += u["line_count"]
                elif u["source"] == "session":
                    if u["id"] not in state["processed_sessions"]:
                        state["processed_sessions"].append(u["id"])
            
            save_state(state)
            print(f"Completato batch {i//batch_size + 1}/{(len(all_new_units)-1)//batch_size + 1}")
        
        except Exception as e:
            print(f"ERRORE nel batch {i//batch_size + 1}: {e}")
            print("Salto il batch e continuo...")
            continue

    # Final count update in file
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            final_content = f.read()
        
        total_processed = sum(state["chatgpt_files"].values()) + state["jsonl_line_count"] + len(state["processed_sessions"])
        final_content = re.sub(r'Sessioni processate: \d+', f'Sessioni processate: {total_processed}', final_content)
        
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write(final_content)
    except: pass

    print(f"Sintesi completata. Memoria aggiornata in {MEMORY_FILE}")

if __name__ == "__main__":
    main()
