# 📚 DocuMind AI

**DocuMind AI** is a document-based **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions based only on the currently active document.

## 🎯 Project Goal

The goal of this project is to build a reliable RAG system that:

- Accepts a PDF document
- Extracts text from the document
- Splits the document into chunks
- Creates embeddings for the chunks
- Stores embeddings in ChromaDB
- Retrieves relevant chunks for each question
- Generates answers using the retrieved context
- Removes old document chunks when a new PDF is processed

## 🔄 How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Answer Generation
```

## 🧠 RAG Pipeline

The application follows these steps:

1. PDF Upload
2. Text Extraction
3. Document Chunking
4. Embedding Generation
5. Vector Storage
6. Similarity Search
7. Context Retrieval
8. Question Answering

## 📄 Active Document Handling

DocuMind AI works with **one active PDF at a time**.

When a new PDF is processed:

- Existing document chunks are removed.
- The new PDF is processed.
- New chunks are created.
- New embeddings are stored in ChromaDB.
- Questions are answered using the current PDF.

This prevents information from a previously processed PDF from affecting answers for the new document.

## ✂️ Chunking

The application uses recursive text splitting for general PDF documents.

Current configuration:

- Chunk size: `800`
- Chunk overlap: `100`

Employee-data PDFs are handled as individual employee records so that employee-specific questions can be answered more precisely.

## 🛠️ Technologies Used

- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- ChromaDB
- Ollama
- Llama 3.2
- Nomic Embed Text
- PyPDF
- LangChain
- Vector Similarity Search

## 📁 Project Structure

```text
RAG_project/
│
├── app.py
├── advanced_rag.py
├── config.py
├── prepare_data.py
├── README.md
├── .gitignore
│
├── documents/
├── data/
└── chroma_store/
```

## ▶️ Run the Project

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

The terminal will display the local Streamlit URL.

## 💬 Example Questions

For an employee information PDF:

```text
What is the salary of EMP001?
How much does EMP001 earn?
Give me who are working in Hyderabad.
What is the manager of EMP001?
What department does EMP001 work in?
```

For a general document:

```text
What should happen when a user enters valid credentials?
What is the purpose of the login functionality?
```

## 🛡️ RAG Behavior

DocuMind AI is designed to answer questions using the currently active document.

Example:

```text
PDF 1
  ↓
Process
  ↓
100 chunks


PDF 2
  ↓
Process
  ↓
Remove old chunks
  ↓
Create new chunks
  ↓
5 chunks
  ↓
Ask questions
  ↓
Answer only from PDF 2
```

This prevents stale information from previously processed documents from being retrieved.

## 🧪 Testing

The application was tested with:

- Employee PDF containing 100 employee records
- QA RAG test PDF containing 5 chunks
- Repeated questions
- Different wording for the same question
- Location-based employee queries
- Switching from one PDF to another

Example:

```text
Employee PDF
→ 100 chunks

QA PDF
→ old 100 chunks removed
→ 5 new chunks indexed
```

Repeated factual questions also return consistent answers.

## 👨‍💻 Author

**Ramesh Guddala**