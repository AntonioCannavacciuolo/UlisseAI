import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import uuid

def migrate():
    project_root = Path(__file__).resolve().parent.parent
    corpus_dir = project_root / "corpus"
    sessions_dir = corpus_dir / "sessions"
    new_conv_file = corpus_dir / "new_conversations.jsonl"
    
    if not new_conv_file.exists():
        print("No new_conversations.jsonl found. Nothing to migrate.")
        return

    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    exchanges = []
    with open(new_conv_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                exchanges.append(json.loads(line))
    
    if not exchanges:
        print("Empty conversations file.")
        return

    # Sort by timestamp
    exchanges.sort(key=lambda x: x.get("timestamp", ""))
    
    sessions = []
    current_session = None
    
    # Proximity threshold: 2 hours
    threshold = timedelta(hours=2)
    
    for exchange in exchanges:
        ts_str = exchange.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str)
        except:
            ts = datetime.now()
            
        if not current_session or (ts - current_session["last_ts"]) > threshold:
            if current_session:
                sessions.append(current_session)
            
            # Start new session
            session_id = str(uuid.uuid4())
            current_session = {
                "id": session_id,
                "title": exchange.get("title", "Migrated Session"),
                "created_at": ts.isoformat(),
                "updated_at": ts.isoformat(),
                "status": "archived",
                "messages": [],
                "last_ts": ts
            }
            
        # Add message pair
        current_session["messages"].append({
            "role": "user",
            "content": exchange.get("user_message", ""),
            "timestamp": ts.isoformat()
        })
        current_session["messages"].append({
            "role": "assistant",
            "content": exchange.get("assistant_response", ""),
            "timestamp": ts.isoformat(),
            "sources": exchange.get("sources", [])
        })
        current_session["updated_at"] = ts.isoformat()
        current_session["last_ts"] = ts

    if current_session:
        sessions.append(current_session)

    for session in sessions:
        # Clean up temp timestamp
        session.pop("last_ts")
        filename = sessions_dir / f"{session['id']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
            
    print(f"Migrated {len(exchanges)} exchanges into {len(sessions)} sessions.")

if __name__ == "__main__":
    migrate()
