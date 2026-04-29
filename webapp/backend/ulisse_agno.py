import os
from pathlib import Path
from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.os import AgentOS
from agno.tools.workspace import Workspace
from agno.models.deepseek import DeepSeek
from agno.tools.websearch import WebSearchTools
from agno.tools.browserbase import BrowserbaseTools
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.tools.local_file_system import LocalFileSystemTools

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
corpus_dir = project_root / os.getenv("CORPUS_PATH", "corpus")
wiki_dir = corpus_dir / "wiki"
wiki_pages_dir = wiki_dir / "pages"

# ==========================================
# 1. Definisci i Tool di Ulisse per Agno
# ==========================================
# In Agno, i tool possono essere semplici funzioni Python con type hints e docstrings.

def wiki_read_page(title: str) -> str:
    """Reads the content of a page from the long-term memory (Wiki)."""
    try:
        import re
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        path = wiki_pages_dir / f"{safe_title}.md"
        if path.exists():
            return f"Content of [[{title}]]:\n\n" + path.read_text(encoding="utf-8")
        return f"The page [[{title}]] does not exist yet."
    except Exception as e:
        return f"Wiki read error: {str(e)}"

def wiki_write_page(title: str, content: str) -> str:
    """Creates or updates a page in the long-term memory (Wiki)."""
    try:
        import re
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        path = wiki_pages_dir / f"{safe_title}.md"
        path.write_text(content, encoding="utf-8")
        return f"Page [[{title}]] updated successfully."
    except Exception as e:
        return f"Wiki write error: {str(e)}"

def wiki_list_pages() -> str:
    """Lists all pages present in the long-term memory."""
    try:
        pages = [f.stem.replace('_', ' ') for f in wiki_pages_dir.glob("*.md")]
        return "Pages present in the Wiki:\n" + "\n".join([f"- [[{p}]]" for p in pages])
    except Exception as e:
        return f"Wiki list error: {str(e)}"


# ==========================================
# 2. Definisci l'Agente Ulisse
# ==========================================
# Prepariamo il system prompt basato su quello esistente
system_prompt_path = corpus_dir / "system_prompt.md"
base_prompt = system_prompt_path.read_text(encoding="utf-8") if system_prompt_path.exists() else "You are Ulisse."

schema_path = wiki_dir / "WIKI_SCHEMA.md"
wiki_schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""

agent_instructions = f"""
{base_prompt}

=== MISSION: WIKI MAINTAINER ===
You are the keeper of Ulisse's long-term memory.
You must decide autonomously (or upon request) when to store important information in the Wiki.
Strictly follow the schema provided below for Wiki management.

{wiki_schema}
"""

# --- Configurazione Toolkits di Agno ---
# Importa qui i toolkit quando richiesto, ad esempio:
# from agno.tools.duckduckgo import DuckDuckGo
# from agno.tools.python import PythonTools

agno_toolkits = [
    WebSearchTools(),      # Abilita la ricerca web (Google, DuckDuckGo, ecc.)
    CsvTools(),            # Abilita la lettura e l'interrogazione di file CSV
    FileTools(base_dir=project_root),  # Abilita la gestione file (lettura/scrittura)
    PythonTools(base_dir=project_root), # Abilita l'esecuzione di codice Python
    ShellTools(base_dir=project_root),  # Abilita l'esecuzione di comandi shell
    LocalFileSystemTools(target_directory=str(project_root)), # Abilita la scrittura organizzata di file
]

# Browserbase è opzionale: si attiva solo se le chiavi sono configurate
browserbase_api_key = os.getenv("BROWSERBASE_API_KEY")
browserbase_project_id = os.getenv("BROWSERBASE_PROJECT_ID")
if browserbase_api_key and browserbase_project_id:
    agno_toolkits.append(
        BrowserbaseTools(
            api_key=browserbase_api_key,
            project_id=browserbase_project_id
        )
    )
    print("✅ BrowserbaseTools attivato (chiavi trovate)")
else:
    print("⚠️ BrowserbaseTools NON attivato: BROWSERBASE_API_KEY e/o BROWSERBASE_PROJECT_ID mancanti. Browser automation non disponibile.")


base_tools = [
    wiki_read_page,
    wiki_write_page,
    wiki_list_pages,
    # Il tool Workspace permette all'agente di leggere e cercare file nel progetto locale
    Workspace(str(project_root), allowed=["read", "list", "search", "write", "edit", "delete", "shell"])
]

# Combina tutti i tools
all_tools = base_tools + agno_toolkits

# Inizializza l'agente Agno
# Utilizziamo OpenAIChat configurato per DeepSeek se necessario, oppure il default openai
ulisse_agent = Agent(
    name="Ulisse",
    model=DeepSeek(
        id="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ),
    description="Ulisse AI - L'assistente con memoria a lungo termine.",
    instructions=agent_instructions,
    tools=all_tools,
    enable_agentic_memory=True,  # Memoria persistente tra sessioni
    add_history_to_context=True,
    num_history_runs=5,
)

# ==========================================
# 3. Avvia AgentOS (Solo se eseguito direttamente)
# ==========================================
if __name__ == "__main__":
    # AgentOS espone l'agente tramite API FastAPI compatibili
    db_path = corpus_dir / "agno_sessions.db"
    agent_os = AgentOS(
        agents=[ulisse_agent],
        tracing=True,
        db=SqliteDb(db_file=str(db_path))
    )

    app = agent_os.get_app()
    # Per eseguire questo file usa:
    # fastapi dev webapp/backend/ulisse_agno.py
