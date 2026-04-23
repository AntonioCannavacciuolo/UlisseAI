# Ulisse Brain

**A persistent memory infrastructure for LLMs. Installable. Customizable. Private.**

---

## What it is

Ulisse is a system that allows a language model to **remember what you told it**, even across different sessions. It's not only a chatbot. It's an architecture that turns a generic LLM into an entity with continuous memory, capable of updating its own knowledge as it talks to you.

It works with:
- Local models (Ollama, LM Studio)
- Remote APIs (OpenAI, Deepseek, Claude, any provider)
- *(Future)* a fine-tuned model exclusive to this architecture

---

## How it works

Ulisse is built on three layers:

1. **Current conversation** — what you're saying now, immediate context.
2. **Vector memory** (RAG) — past conversations, uploaded documents, notes. Retrieves what's relevant to the current question.
3. **Synthetic memory** — a structured document summarizing everything you've shared, updated automatically. Like ChatGPT's memory, but fully yours.

The user doesn't need to configure any of this. It works out of the box.

---

## Stack

- Python 3.11
- ChromaDB (vector database)
- Deepseek API (or any OpenAI-compatible provider)
- Obsidian vault (visual knowledge graph)
- Node.js webapp (frontend interface)

---

## Main flow

```
JSON export → scripts → ChromaDB vectordb → Deepseek API → webapp
```

---

## Installation

Requirements:
- Python 3.11+
- An LLM API key (Deepseek, OpenAI, or compatible)

```bash
git clone https://github.com/AntonioCannavacciuolo/UlisseAI.git
cd UlisseAI
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API key:

```
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
CORPUS_PATH=./corpus
VAULT_PATH=./vault
VECTORDB_PATH=./vectordb
```

Run the backend:

```bash
python webapp/backend/app.py
```

Open your browser at `http://localhost:5000`.

---

## Windows Services (NSSM)

To run the backend and sync watcher as persistent Windows Services:

1. Ensure [NSSM](https://nssm.cc/) is installed at `C:\nssm-2.24\win64\nssm.exe`.
2. Run `powershell .\install_services.ps1` as **Administrator**.
3. Manage services via `services.msc` (look for `Ulisse Brain - Backend` and `Ulisse Brain - Sync Watcher`).

To remove services:

```powershell
powershell .\uninstall_services.ps1
```

---

## Philosophy

Ulisse exists for three reasons:

1. **Memory shouldn't be optional.** Chatbots forget everything after a few messages. Ulisse is built to remember.
2. **Your data is yours.** Every instance is local. No central server. No extraction.
3. **Control belongs to the user.** You choose the model, the prompt, the data to load. No black boxes.

---

## Project status

Currently in active development. It works, but is still in testing phase. The interface is minimal. Contributions, bug reports, and ideas are welcome.

---

## License

*(to be chosen — MIT, Apache 2.0, AGPL?)*

---

## Contact

*(link to GitHub Issues, eventual community)*
