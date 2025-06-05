# ArtRAGSys

A Multimodal Retrieval-Augmented Generation (RAG) system for art collections, featuring a modern chat-like GUI and advanced search capabilities.

Project for the 2025 course "Information Extraction and Retrieval for Multilingual Natural Language Data" (FH Campus Wien, Multilingual Technologies MSc).

---

## Features
- **Modern GUI**: Built with CustomTkinter, supports dark/light/system themes.
- **Multimodal Search**: Search artworks by text, semantics, metadata, or hybrid methods.
- **Chat with AI**: Ask questions about artworks and get context-aware answers from an LLM (Ollama backend).
- **Database**: Uses SQLite for metadata and ChromaDB for vector search.
- **Easy Setup**: Databases are prebuilt and included in the repository.

---

## Installation & Setup

### 1. Clone the repository
```sh
git clone <repo-url>
cd ArtRAGSys
```

### 2. Install Python dependencies
It is recommended to use a virtual environment:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download NLTK and spaCy data
```sh
python -m nltk.downloader punkt
python -m spacy download en_core_web_trf
```

### 4. Install Ollama (for LLM backend)
Follow the instructions for your OS at: https://ollama.com/download

For macOS, you can use Homebrew:
```sh
brew install ollama
```

Or download directly from the website.

### (Optional) 5. Install ArtRAGSys in editable mode
If you want to use the CLI commands (e.g., `artrag-gui`) and have code changes reflected immediately, install the package in editable mode:
```sh
pip install -e .
```

---

## Running the GUI

Make sure Ollama is running with the required model (e.g., `gemma3:4b-it-qat`).

```sh
python -m src.gui_modern
```

Or, if installed as a package:
```sh
artrag-gui
```

---

## Project Structure
- `src/` — Main source code (GUI, database, retrieval, Ollama integration)
- `data/` — Art images and CSV data
- `requirements.txt` — Python dependencies
- `setup.py` — Installable package setup

---

## Requirements
- Python 3.8+
- Ollama (for LLM backend, see https://ollama.com/)
- [Optional] CUDA-enabled GPU for faster embedding/model inference

---

## License
MIT License

---

## Acknowledgements
- FH Campus Wien, Multilingual Technologies MSc
- Open-source libraries: CustomTkinter, ChromaDB, SentenceTransformers, spaCy, NLTK, Pillow, etc.

---

## Git Maintenance: Pruning Unreachable Objects

To clean up unreachable Git objects and free up space, run:

```sh
git gc --prune=now
```

If you use Git LFS, you can also remove old LFS files with:

```sh
git lfs prune
```

**Warning:** These commands permanently remove data that is no longer referenced by any commit.

