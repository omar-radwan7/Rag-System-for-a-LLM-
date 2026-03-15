# RAG System for an LLM

A full Retrieval-Augmented Generation (RAG) pipeline built in Python, designed to enhance LLM responses with document-grounded context. The system retrieves relevant information from a knowledge base using vector similarity search and injects it into the LLM prompt, significantly improving factual accuracy and reducing hallucinations.

---

## Overview

Large Language Models often produce confident but incorrect answers when asked about domain-specific or up-to-date information. This project addresses that limitation by implementing a RAG pipeline that:

1. Ingests and chunks documents into a vector store
2. Embeds user queries and retrieves the most semantically relevant chunks
3. Augments the LLM prompt with the retrieved context
4. Evaluates response quality using automated metrics

---

## Features

- **Document Ingestion** — Processes and chunks raw documents for vector storage
- **Semantic Retrieval** — Embeds queries and performs similarity search to find relevant context
- **LLM Integration** — Feeds retrieved context into the LLM for grounded responses
- **Evaluation Pipeline** — Measures retrieval and generation quality using automated metrics
- **Streamlit UI** — Interactive web interface for querying the system
- **Survey & Results** — Structured evaluation results and benchmarking data
- **Configurable** — JSON-based configuration for models, chunking, and retrieval parameters

---

## Project Structure

```
Rag-System-for-an-LLM/
├── rag/                  # Core RAG pipeline (ingestion, retrieval, generation)
├── eval/                 # Evaluation scripts and metrics
├── results/              # Benchmark results and output logs
├── survey/               # Survey data used for evaluation
├── ui/                   # Streamlit frontend
├── data/                 # Input documents and knowledge base
├── config.json           # Pipeline configuration
├── eval_output.log       # Evaluation run logs
├── streamlit_output.log  # UI run logs
└── README.md
```

---

## Tech Stack

| Component        | Technology                        |
|-----------------|-----------------------------------|
| Language         | Python                            |
| LLM Interface    | OpenAI API / local LLM            |
| Embeddings       | Sentence Transformers             |
| Vector Store     | FAISS / ChromaDB                  |
| UI               | Streamlit                         |
| Evaluation       | Custom metrics (precision, recall, faithfulness) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/omar-radwan7/Rag-System-for-an-LLM-.git
cd Rag-System-for-an-LLM-
pip install -r requirements.txt
```

### Configuration

Edit `config.json` to set your model, chunking strategy, and retrieval parameters:

```json
{
  "model": "gpt-3.5-turbo",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 5
}
```

### Run the UI

```bash
streamlit run ui/app.py
```

### Run Evaluation

```bash
python eval/evaluate.py
```

Results will be saved to `results/` and logged in `eval_output.log`.

---

## How It Works

```
User Query
    │
    ▼
Embed Query (Sentence Transformers)
    │
    ▼
Vector Similarity Search (FAISS / ChromaDB)
    │
    ▼
Retrieve Top-K Relevant Chunks
    │
    ▼
Augment LLM Prompt with Context
    │
    ▼
Generate Grounded Response
    │
    ▼
Evaluate (Faithfulness, Relevance, Recall)
```

---

## Evaluation

The pipeline includes an automated evaluation framework that measures:

- **Retrieval Precision** — How relevant are the retrieved chunks?
- **Answer Faithfulness** — Does the response stay grounded in the retrieved context?
- **Answer Relevance** — Does the response address the original question?

Evaluation results are stored in `results/` for benchmarking and comparison.

---

## License

MIT License
