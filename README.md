# Ulisse Brain

**A persistent memory infrastructure for LLMs. Installable. Customizable. Private.**

---

## What it is

Ulisse is a system that allows a language model to **remember what you told it**, even across different sessions. It's not only a chatbot. It's an architecture that turns a generic LLM into an entity with continuous memory, capable of updating its own knowledge as it talks to you.

It works with:
- **Local models** (Ollama, LM Studio, any OpenAI-compatible server)
- **Remote APIs** (OpenAI, DeepSeek, Claude, any provider with an API key)
- **Ulisse Memo v1** — a fine-tuned cloud model specifically built for this architecture

---

## How it works

Ulisse is built on three layers:

1. **Current conversation** — Immediate context of the chat.
2. **Short-Term Memory (STM - RAG)** — Context-aware retrieval from past sessions and documents using ChromaDB.
3. **Long-Term Memory (LTM - LLM Wiki)** — An AI-managed semantic Wiki (Markdown) where Ulisse synthesizes and organizes persistent knowledge, projects, and facts.

The user doesn't need to configure any of this. It works out of the box.

---

## Capabilities & Toolkits

Ulisse is not just a chatbot; it is an **agentic system** equipped with a specialized Agno sub-agent and native tools to interact with the real world.

### 🛠️ Native Tools
Directly built into the core for maximum reliability:
- **Workspace Reader:** List files and folders in your project (`native_list_files`).
- **File Reader:** Open and read any file within the workspace (`native_read_file`).
- **Wiki Management:** Read, write, and organize the semantic long-term memory.

### 🤖 Agno Agent (Sub-Agent)
A powerful autonomous engine for complex tasks:
- **Web Search:** Browse the internet for real-time information.
- **Python Runner:** Write and execute Python scripts to solve complex logic or math.
- **Shell Access:** Execute terminal commands directly in the workspace.
- **CSV Analysis:** Query large CSV datasets using SQL (via DuckDB).
- **Browser Automation:** Advanced web scraping and interaction (requires optional [Browserbase](https://www.browserbase.com/) API keys).

---

## Stack

- Python 3.11+
- ChromaDB (vector database for STM)
- DeepSeek API or any OpenAI-compatible provider
- Semantic Wiki (Markdown-based LTM)
- Obsidian-compatible knowledge graph
- Flask + Vanilla JS Frontend

---

## Main flow

```
Chat Interaction ↔ AI Tool Calling ↔ Semantic Wiki (LTM)
       ↕
  ChromaDB (STM)
```

---

## Installation

Requirements:
- Python 3.11+
- An LLM API key **OR** Ollama/LM Studio running locally **OR** use Ulisse Memo v1 (no key needed)

```bash
git clone https://github.com/AntonioCannavacciuolo/UlisseAI.git
cd UlisseAI
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your settings:

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

## Choosing the AI Model (UI)

In the chat interface, next to the **Send** button you'll find a 🌐 **network button**.  
Clicking it opens a dropdown with three options:

| Option | Description |
|--------|-------------|
| 💻 **LLM Locale** | Connects to a model running locally on your machine (Ollama, LM Studio, etc.). Uses the `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY` from `.env`. |
| 🔑 **API Key** | Lets you enter any provider's Base URL, API Key, and model name directly from the UI. Credentials are saved in your browser's `localStorage`. |
| 🔮 **Ulisse Memo v1** | Connects to the Ulisse Memo cloud model at `https://ulisse-memo.onrender.com`. No API key required. |

Your choice is **persisted in the browser** across reloads.

---

## Changing the Model Manually (Advanced)

### Option A — Local / Default model

Edit `.env`:

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com   # or http://localhost:11434/v1 for Ollama
```

Then in `webapp/backend/app.py`, find the line:

```python
chat_model  = "deepseek-chat"
```

and change it to the model name you want (e.g. `"gpt-4o"`, `"llama3"`, `"qwen2.5:7b"`).

### Option B — Ulisse Memo v1 endpoint

By default the app connects to:

```
https://ulisse-memo.onrender.com
```

### Option C — Fully custom provider in code

In `webapp/backend/app.py` locate the provider routing block (~line 408):

```python
# === Provider routing ===
provider = data.get("provider", "local")
```

You can add new branches here to support additional providers at the server level.

---

## Philosophy

Ulisse exists for three reasons:

1. **Memory shouldn't be optional.** Chatbots forget everything after a few messages. Ulisse is built to remember.
2. **Your data is yours.** Every instance is local. No central server. No extraction.
3. **Control belongs to the user.** You choose the model, the prompt, the data to load. No black boxes.

---

## Project status

Currently in active development. It works, but is still in testing phase. Contributions, bug reports, and ideas are welcome.

---

## License

*Apache 2.0*

---

## Contact

*(link to GitHub Issues, eventual community)*
