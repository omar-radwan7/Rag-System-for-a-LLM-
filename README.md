# RAG System for an LLM

A full Retrieval-Augmented Generation (RAG) pipeline built in Python, designed to enhance LLM responses with document-grounded context. The system runs entirely locally using **Ollama** for LLM inference, retrieves relevant information from a knowledge base using vector similarity search, and injects it into the prompt—significantly improving factual accuracy and reducing hallucinations.

---

## Overview

Large Language Models often produce confident but incorrect answers when asked about domain-specific or up-to-date information. This project addresses that limitation by implementing a RAG pipeline that:

1. Ingests and chunks documents into a vector store
2. Embeds user queries and retrieves the most semantically relevant chunks
3. Augments the LLM prompt with the retrieved context using a **locally-running Ollama model**
4. Evaluates response quality using automated metrics

### Why Local with Ollama?

- **No API costs** — Run inference completely offline, no external API calls
- **Privacy** — Your documents and queries never leave your machine
- **Control** — Choose your model, adjust parameters, debug freely
- **Speed** — Latency depends only on hardware, not network

---

## Features

- **Document Ingestion** — Processes and chunks raw documents for vector storage
- **Semantic Retrieval** — Embeds queries and performs similarity search to find relevant context
- **Local LLM Integration** — Uses Ollama to run open-source models (Mistral, Llama 2, Neural Chat, etc.) for grounded response generation
- **Evaluation Pipeline** — Measures retrieval and generation quality using automated metrics (precision, recall, faithfulness)
- **Streamlit UI** — Interactive web interface for querying the system with real-time responses
- **Survey & Results** — Structured evaluation results and benchmarking data
- **Configurable** — JSON-based configuration for models, chunking, retrieval parameters, and Ollama settings

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
├── config.json           # Pipeline configuration (models, chunking, Ollama settings)
├── eval_output.log       # Evaluation run logs
├── streamlit_output.log  # UI run logs
└── README.md
```

---

## Tech Stack

| Component        | Technology                                    |
|-----------------|-----------------------------------------------|
| Language         | Python 3.9+                                   |
| Local LLM        | **Ollama** (Mistral, Llama 2, Neural Chat)   |
| Embeddings       | Sentence Transformers (all-MiniLM-L6-v2)    |
| Vector Store     | FAISS / ChromaDB                              |
| UI               | Streamlit                                     |
| Evaluation       | Custom metrics (precision, recall, faithfulness) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- **Ollama** installed and running ([download here](https://ollama.ai))

### Installation

```bash
# Clone the repository
git clone https://github.com/omar-radwan7/Rag-System-for-an-LLM-.git
cd Rag-System-for-an-LLM-

# Install Python dependencies
pip install -r requirements.txt
```

### Ollama Setup

1. **Install Ollama** from [ollama.ai](https://ollama.ai)
2. **Pull a model** (choose one):
   ```bash
   ollama pull mistral      # Fast, good quality
   ollama pull llama2       # Larger, more capable
   ollama pull neural-chat  # Optimized for chat
   ```
3. **Start Ollama server** (runs on `http://localhost:11434` by default):
   ```bash
   ollama serve
   ```

### Configuration

Edit `config.json` to set your model, chunking strategy, retrieval parameters, and Ollama settings:

```json
{
  "ollama_model": "mistral",
  "ollama_base_url": "http://localhost:11434",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 5,
  "temperature": 0.7,
  "max_tokens": 256
}
```

### Run the UI

```bash
streamlit run ui/app.py
```

Open your browser to `http://localhost:8501` and start querying your documents.

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
Generate Response via Ollama (Local)
    │
    ▼
Evaluate (Faithfulness, Relevance, Recall)
```

---

## Example Usage

### Via Streamlit UI

1. Streamlit loads your documents from `data/`
2. Enter a query: *"What are the key benefits of the pension plan?"*
3. The system retrieves relevant chunks and generates a grounded response
4. View retrieved context alongside the answer

### Via Python API

```python
from rag.pipeline import RAGPipeline

# Initialize pipeline with Ollama
pipeline = RAGPipeline(config_path="config.json")

# Query the system
query = "What are the key benefits?"
response = pipeline.query(query)

print(f"Answer: {response['answer']}")
print(f"Retrieved chunks: {response['context']}")
print(f"Evaluation scores: {response['scores']}")
```

---

## Evaluation

The pipeline includes an automated evaluation framework that measures:

- **Retrieval Precision** — How relevant are the retrieved chunks to the query?
- **Answer Faithfulness** — Does the response stay grounded in the retrieved context (no hallucinations)?
- **Answer Relevance** — Does the response directly address the original question?

Evaluation results are stored in `results/` for benchmarking across different models and configurations.

---

## Performance Considerations

| Aspect | Local Ollama | Cloud API |
|--------|-------------|-----------|
| **Cost** | Free | Per-token billing |
| **Privacy** | On-device only | Sent to provider |
| **Speed** | Hardware-dependent | Network + processing |
| **Control** | Full (model choice, params) | Limited to API options |
| **Offline** |  Yes |  Requires internet |

For production use, start with `mistral` (fast, 7B params) or `neural-chat` (optimized, 7B). Use `llama2` (13B) for higher quality if hardware allows.

---

## Troubleshooting

**Ollama connection refused?**
```bash
# Make sure Ollama is running
ollama serve
```

**Model not found?**
```bash
# Pull the model first
ollama pull mistral
```

**Out of memory?**
- Use a smaller model: `ollama pull neural-chat` instead of `llama2`
- Reduce `chunk_size` in config.json
- Lower `max_tokens` in generation settings

---

## License

MIT License
