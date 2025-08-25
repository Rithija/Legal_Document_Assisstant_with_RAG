# ⚖️ Legal_Document_Assisstant_with_RAGg

> An intelligent RAG-based application that answers legal questions by retrieving relevant information from 10 years of Indian Supreme Court judgments.

---

## 🎥 Demo & Screenshots





`![Application Screenshot](assets/app_screenshot1.png)`
`![Application Screenshot](assets/app_screenshot1.png)`

---

## ✨ Features

* **Natural Language Queries:** Ask complex legal questions in plain English.
* **Contextual Answers:** The system provides answers synthesized directly from the content of real court judgments.
* **Source Verification:** Every answer is backed by citations and snippets from the original source documents, allowing for easy verification.
* **High Performance:** Powered by the ultra-fast Groq LPU™ Inference Engine for real-time responses.
* **Comprehensive Knowledge Base:** Built on a processed and vectorized dataset of over 53,000 text chunks from 10 years of Indian Supreme Court judgments.

---

## 🛠️ Tech Stack

* **Backend & Orchestration:** Python, LangChain
* **Frontend:** Streamlit
* **Vector Database:** ChromaDB
* **Embedding Model:** `all-mpnet-base-v2` (Sentence-Transformers)
* **LLM:** Llama 3 (via Groq API)
* **Data Processing:** PyPDF, Pandas, Scikit-learn

---

## 🏛️ High-Level Architecture

The project is divided into two main pipelines: a one-time Data Processing Pipeline and a real-time RAG Application Pipeline.



[Image of the System Architecture Diagram]

`![Architecture Diagram](assets/architecture.png)`

#### **Phase 1: Data Ingestion & Embedding (One-Time Pipeline)**

This pipeline processes the raw PDF judgments and converts them into a queryable vector database.

`(PDFs) -> [Data Processing Script] -> (Clean Chunks) -> [Embedding Script] -> (Vector Database - ChromaDB)`

#### **Phase 2: RAG Application (Real-Time Pipeline)**

This is the live application that interacts with the user.

`(User Query) -> [Streamlit App] -> (Vectorized Query) -> [ChromaDB] -> (Relevant Chunks) -> [Groq LLM] -> (Synthesized Answer) -> [Streamlit App]`

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.10+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/Rithija/Legal_Document_Assisstant_with_RAG.git)
    cd [Your-Repo-Name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Groq API Key:**
    Create a `.env` file in the root directory and add your API key:
    ```
    GROQ_API_KEY="your_api_key_here"
    ```
    *Alternatively, you can set it as an environment variable:*
    ```bash
    export GROQ_API_KEY="your_api_key_here"
    ```

### Running the Project

1.  **Prepare the Data:**
    * Place your PDF judgments inside the `data/` directory (or follow the structure from the Kaggle dataset).
    * Run the processing script to generate the chunks:
        ```bash
        python process_data.py
        ```
    * Run the embedding script to create the vector database. **(This requires a CUDA-enabled GPU)**:
        ```bash
        python embed_and_store.py
        ```

2.  **Launch the Application:**
    ```bash
    streamlit run app.py
    ```
    Open your browser to the local address provided by Streamlit.

---

## 📂 Project Structure

```
.
├── app.py                  # The main Streamlit application script
├── process_data.py         # Script for processing and chunking PDFs
├── embed_and_store.py      # Script for creating embeddings and storing in ChromaDB
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to be ignored by Git
├── assets/                 # For storing images and diagrams
│   ├── architecture.png
|   ├── app_screenshot1.png
│   └── app_screenshot2.png
|
├── data/                   # (Ignored by Git) For raw PDF files
├── chroma_db/              # (Ignored by Git) The persistent vector database
└── processed_chunks.pkl    # (Ignored by Git) Intermediate processed data
```

