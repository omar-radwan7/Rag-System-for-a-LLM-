BookMind RAG: Local AI Book Assistant
=======================================

BookMind is a high-performance, Retrieval-Augmented Generation (RAG) system designed to turn any PDF book into an interactive, knowledgeable companion. Built with a privacy-first philosophy, the entire pipeline runs locally using Ollama, ensuring your documents and conversations never leave your machine.

Whether you're exploring the built-in knowledge base or uploading your own collection, BookMind provides grounded, accurate answers with deep context awareness and multilingual support (optimized for Arabic and English).

---

Key Features
------------

- Dynamic PDF Support: Upload any PDF book directly through the UI. The system automatically extracts text, chunks content, and builds an in-memory vector index in seconds.
- Local-First Architecture: Powered by Ollama. No API keys, no subscription costs, and 100% data privacy.
- Multilingual Mastery: Specialized support for Arabic and English, featuring a custom-designed Streamlit UI with Noto Naskh Arabic typography and RTL support.
- 3-Way Context Logic:
    1. In-Context: Answers directly from the book's text.
    2. General Knowledge: Answers book-related questions using its internal training if the context is missing.
    3. Refusal Logic: Politely declines unrelated topics (e.g., cooking or sports) to stay focused on the book.
- Real-time Feedback: Streaming responses for a natural chat experience, with transparent latency tracking for both retrieval and generation.
- Source Transparency: Every answer includes clickable source chips showing exactly which chunks were retrieved from the text.
- Chat History: Persistent session management—save, rename, and return to your conversations later.

---

Tech Stack
----------

| Component | Technology |
| --- | --- |
| Language | Python 3.9+ |
| LLM Inference | Ollama (Mistral, Llama 3, Qwen 2.5) |
| Embeddings | intfloat/multilingual-e5-small |
| Vector Database | FAISS (FlatIP for high-precision similarity) |
| PDF Processing | pypdf |
| Frontend | Streamlit (Custom Glassmorphic CSS) |
| Logic | Sentence Transformers & NumPy |

---

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~
- Ollama installed and running.
- Python 3.9 or higher.

Installation
~~~~~~~~~~~~
```bash
# Clone the repository
git clone https://github.com/omar-radwan7/Rag-System-for-an-LLM-.git
cd Rag-System-for-an-LLM-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Model Setup
~~~~~~~~~~~
Pull the recommended model for Arabic/English performance:
```bash
ollama pull qwen2.5:3b  # Fast and highly capable for multilingual tasks
```

Run the App
~~~~~~~~~~~
```bash
streamlit run ui/app.py
```

---

Project Structure
-----------------

```text
Rag-System-for-an-LLM/
├── rag/                 # Core Intelligence
│   ├── pdf_indexer.py   # In-memory PDF processing & FAISS building
│   ├── rag_pipeline.py  # Prompt engineering & retrieval logic
│   └── ollama_client.py # Ollama API integration
├── ui/                  # Frontend
│   ├── app.py           # Streamlit Chat Interface
│   └── chat_storage.json# Persistent session data
├── eval/                # Quality Assurance
│   ├── eval_runner.py   # Automated benchmarking script
│   └── gold_questions.json # Ground-truth test set
├── data/                # Sample dataset & cleaning scripts
├── results/             # Performance logs & evaluation reports
└── config.json          # System-wide settings (models, top_k, etc.)
```

---

Evaluation and Benchmarking
---------------------------

BookMind includes a dedicated evaluation framework to ensure accuracy and minimize hallucinations. The eval_runner.py script measures:
- Retrieval Accuracy: How well the system finds relevant chunks.
- Response Latency: End-to-end timing for retrieval vs. generation.
- Faithfulness: Consistency between the retrieved context and the generated answer.

Run evaluation:
```bash
python eval/eval_runner.py
```

---

UI Customization
----------------
The interface is optimized for a premium reading experience:
- Glassmorphic Sidebar: Clean, transparent design for chat history.
- Typography: Uses Noto Naskh Arabic for beautiful Arabic rendering.
- RTL/LTR Support: Automatically adapts based on the message language.

---

License
-------
MIT License
