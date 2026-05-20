BookMind RAG: Local Multilingual Knowledge Retrieval System
=======================================================

Overview and Academic Vision
----------------------------

BookMind is a high-performance, Retrieval-Augmented Generation (RAG) system designed to transform static document repositories—specifically PDFs —into interactive, context-aware knowledge bases. This project explores the intersection of data privacy, local Large Language Model (LLM) inference, and multilingual information retrieval.

Developed as part of an intensive research and development initiative at the Applied Innovation Center (AIC), the system addresses the critical need for organizations to interact with sensitive internal data without compromising data sovereignty. The core of this project serves as a foundational prototype for a graduation thesis focusing on localized AI solutions for complex linguistic environments.

Problem Statement
-----------------

Modern Large Language Models, while powerful, face three primary challenges when applied to specialized domains:

1. Knowledge Cutoffs and Hallucinations: Standard LLMs are limited to their training data and often produce factually incorrect responses (hallucinations) when asked about specific or niche documents not included in their pre-training.
2. Data Privacy and Sovereignty: Cloud-based AI solutions require uploading sensitive documents to external servers, which is often prohibited by institutional security policies or legal regulations (e.g., GDPR).
3. Linguistic Limitations: Many commercial models struggle with the nuances of specific dialects, such as Egyptian Colloquial Arabic (ECA), often defaulting to Modern Standard Arabic (MSA) or English, which can lead to reduced engagement and accuracy in localized contexts.

The Proposed Solution
---------------------

BookMind solves these issues by implementing a fully local RAG pipeline. By decoupling the knowledge base from the model's internal weights, the system ensures that:

- Information is retrieved directly from the provided source (PDF), eliminating hallucinations regarding document content.
- All processing occurs on local hardware via Ollama, ensuring zero data leakage to external providers.
- The system is optimized for multilingual retrieval, supporting MSA, ECA, and English through specialized embedding models and custom prompt engineering.

Key Technical Features
----------------------

Dynamic In-Memory Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike traditional RAG systems that rely on static, disk-based vector stores, BookMind implements a dynamic indexing layer. When a user uploads a PDF, the system performs real-time text extraction, recursive character chunking, and vector embedding. These vectors are stored in an in-memory FAISS (Facebook AI Similarity Search) index, allowing for immediate interaction without the overhead of database management or persistent storage cleanup.

Advanced Multilingual Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A primary research focus of this project is the handling of Arabic dialects. The system uses the "intfloat/multilingual-e5-small" embedding model, which is highly effective at mapping semantic relationships across different languages and dialects. This enables the system to understand queries in Egyptian Colloquial Arabic and retrieve relevant context even if the source document is written in Modern Standard Arabic.

Three-Way Context Logic
~~~~~~~~~~~~~~~~~~~~~~~
The system employs a sophisticated prompt engineering strategy to manage model behavior:
1. Grounded Retrieval: Priority is given to information found within the retrieved document chunks.
2. Domain-Specific Supplementation: If the answer is not explicitly in the text but the question is relevant to the book's domain, the model uses its internal knowledge to provide a helpful response.
3. Strict Out-of-Scope Refusal: To maintain the system's integrity as a "Book Assistant," it is programmed to politely decline queries unrelated to the book's content or domain.

Technical Architecture
----------------------

The pipeline is structured into three distinct layers:

1. Ingestion Layer: Uses pypdf for robust text extraction and a sliding-window chunking algorithm to preserve semantic context across chunk boundaries.
2. Retrieval Layer: Leverages FAISS for L2 or Inner Product similarity search. It includes a query expansion module that resolves pronouns and follow-up references using conversation history.
3. Generation Layer: Utilizes the Ollama REST API to interface with state-of-the-art open-source models like Mistral, Llama 3, and Qwen 2.5.

Thesis and Research Value
-------------------------

This project provides a platform for investigating several research questions:

- Efficiency vs. Accuracy: How does chunk size and overlap impact retrieval precision in long-form Arabic texts?
- Local Inference Benchmarking: Analyzing the trade-offs between model quantization (e.g., 4-bit) and the quality of grounded responses on consumer-grade hardware.
- Dialectal Robustness: Measuring the system's ability to bridge the gap between colloquial queries and formal source materials.

Project Structure
-----------------

Rag-System-for-an-LLM/
├── rag/                 # Core Intelligence and RAG Pipeline
│   ├── pdf_indexer.py   # In-memory PDF processing and FAISS indexing
│   ├── rag_pipeline.py  # Prompt engineering, retrieval, and query expansion
│   └── ollama_client.py # Lightweight wrapper for Ollama API
├── ui/                  # Frontend Interface
│   ├── app.py           # Streamlit Chat Interface with Arabic RTL support
│   └── chat_storage.json# Persistent session management
├── eval/                # Evaluation Framework
│   ├── eval_runner.py   # Performance and accuracy benchmarking
│   └── gold_questions.json # Standardized test set for research validation
├── data/                # Dataset management and preprocessing scripts
├── results/             # Performance logs and research outputs
└── config.json          # System-wide configuration settings

Getting Started
---------------

Prerequisites
- Ollama installed and running (https://ollama.ai).
- Python 3.9 or higher.

Installation
1. Clone the repository.
2. Create a virtual environment: python -m venv venv.
3. Install dependencies: pip install -r requirements.txt.

Model Configuration
The system is optimized for the Qwen 2.5 3B model due to its exceptional multilingual performance.
Pull the model: ollama pull qwen2.5:3b.

Execution
Run the Streamlit interface: streamlit run ui/app.py.

License
-------
MIT License
