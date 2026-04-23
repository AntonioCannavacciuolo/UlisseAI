# Ulisse Brain

**Purpose:** AI that wants to gain more knowledge, fed by a knowledge corpus

## Stack
- Python 3.11
- ChromaDB
- Deepseek API
- Obsidian vault
- Node.js webapp

## Main flow
ChatGPT JSON export → scripts → ChromaDB vectordb → Deepseek API → webapp

## Windows Services (NSSM)
To run the backend and sync watcher as Windows Services:
1. Ensure [NSSM](https://nssm.cc/) is installed at `C:\nssm-2.24\win64\nssm.exe`.
2. Run `powershell .\install_services.ps1` as **Administrator**.
3. Manage services via `services.msc` (Look for `Ulisse Brain - Backend` and `Ulisse Brain - Sync Watcher`).

To remove services:
1. Run `powershell .\uninstall_services.ps1` as **Administrator**.
