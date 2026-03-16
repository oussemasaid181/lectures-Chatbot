# 📚 PDF RAG Pipeline

A local Retrieval-Augmented Generation (RAG) pipeline that lets you query your PDF documents using natural language. Built with LangChain, ChromaDB, and HuggingFace.

---

## How It Works

```
Your PDFs
    ↓  (ingest)
Split into chunks → stored in Chroma with embeddings
    ↓  (query)
Your question → find similar chunks → LLM generates answer
```

---

## Requirements

- Python 3.11 or lower (Python 3.13 has compatibility issues with some dependencies)
- pip

---

## Installation

```bash
# Clone the repo and navigate into it
cd langchain

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install langchain langchain-community langchain-chroma langchain-huggingface
pip install sentence-transformers chromadb pypdf python-dotenv
pip install transformers accelerate
```

---

## Setup

1. Add your PDFs to the `data/info3/` folder
2. Create a `.env` file in the root:

```env
HF_TOKEN=your_huggingface_token_here
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Usage

### Step 1 — Ingest (run once)

Loads your PDFs, splits them into chunks, and saves embeddings to ChromaDB:

```bash
python script.py ingest
```

Re-run this whenever you:
- Add new PDFs to `data/info3/`
- Change the embedding model
- Change `chunk_size` or `chunk_overlap`

### Step 2 — Query

Ask questions about your documents:

```bash
python script.py query "your question here"
```

**Examples:**

```bash
python script.py query "What is a finite state machine?"
python script.py query "Wie funktioniert Round-Robin Scheduling?"
python script.py query "What are the shortcomings of FSMs?"
```

---

## Configuration

Key settings at the top of `script.py`:

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/info3` | Folder containing your PDFs |
| `CHROMA_PATH` | `chroma` | Where the vector database is stored |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model used to embed chunks |
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `200` | Overlap between chunks |

---

## Models

### Embedding Model
Used to convert text chunks into vectors for similarity search:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```
For multilingual PDFs (e.g. German), switch to:
```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### LLM (Answer Generation)

The project supports HuggingFace models for generating answers. Recommended options:

| Model | Size | Notes |
|---|---|---|
| `facebook/opt-350m` | 500MB | Fast, basic quality |
| `microsoft/phi-2` | 5GB | Better quality, needs more RAM |
| `mistralai/Mistral-7B-Instruct-v0.3` | API | Best quality, via HF Inference API (free) |

To use the HuggingFace Inference API (recommended — no download needed):
```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)
```

---

## Project Structure

```
langchain/
├── data/
│   └── info3/          # Put your PDFs here
├── chroma/             # Auto-generated vector database (after ingest)
├── venv/               # Virtual environment
├── script.py           # Main script
├── .env                # API keys (never commit this)
└── README.md
```

---

## Tips

- **RAG retrieves, it doesn't summarize** — ask specific questions, not "summarize the document"
- **Query in the same language as your PDFs** for best results
- **Small models hallucinate** — use the HuggingFace Inference API for reliable answers
- **Re-ingest after any changes** to PDFs or chunking settings
