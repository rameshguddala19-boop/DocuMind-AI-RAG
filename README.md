# 📚 DocuMind AI RAG

DocuMind AI RAG is a document-based question answering application that allows users to upload a PDF and ask questions based only on the currently active document.

## 🎯 Project Goal

The main goal of this project is to build a reliable RAG (Retrieval-Augmented Generation) system that:

* Accepts a PDF document from the user
* Extracts text from the PDF
* Splits the document into meaningful chunks
* Creates embeddings for the chunks
* Stores the embeddings in a vector database
* Retrieves relevant chunks for each question
* Generates answers using the retrieved document context
* Prevents information from an old PDF from affecting a newly uploaded PDF

## 🔄 How It Works

```text
PDF Upload
    ↓
Extract Text
    ↓
Split into Chunks
    ↓
Generate Embeddings
    ↓
Store in Chroma
    ↓
User Question
    ↓
Retrieve Relevant Chunks
    ↓
Generate Answer
```

## 🧠 RAG Pipeline

The application follows these main steps:

1. PDF Upload
2. Text Extraction
3. Document Chunking
4. Embedding Generation
5. Vector Storage
6. Similarity Search
7. Context Retrieval
8. Question Answering

## 📄 Active Document Handling

The application is designed to work with one active PDF at a time.

When a new PDF is uploaded:

* The previous document's indexed chunks are removed.
* The new PDF is processed.
* New chunks are created and indexed.
* Questions are answered using the currently active document.

This prevents stale information from previously uploaded documents from being used in new answers.

## ✂️ Chunking

The document is divided into chunks before embedding.

Current chunking configuration:

* Chunk size: `800`
* Chunk overlap: `120`

Separators are used to preserve meaningful text boundaries where possible.

## 🗂️ Project Structure

```text
RAG_project/
│
├── app.py
├── advanced_rag.py
├── config.py
├── prepare_data.py
├── rag.db
│
├── documents/
├── data/
├── chroma_store/
├── live_dynamic_rag/
│
├── .env
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

* Python
* Streamlit
* RAG
* ChromaDB
* PDF text extraction
* Embeddings
* Vector similarity search
* SQLite
* Ollama

## ▶️ Run the Project

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the Streamlit application:

```powershell
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 💬 Example Usage

1. Open the application.
2. Upload a PDF.
3. Wait until the document is indexed.
4. Ask a question about the uploaded PDF.
5. The application retrieves relevant chunks.
6. The answer is generated from the active document.

## 🔍 Example Questions

For an employee information PDF:

```text
What is the casual leave policy?
Who has the highest salary in Hyderabad?
Which employee works as a Product Manager?
```

For another PDF, questions should be based only on that newly uploaded document.

## 🛡️ Important RAG Behavior

The application should not answer a question using chunks belonging to a previously uploaded PDF.

For example:

```text
PDF 1 → Upload → Index → Ask Questions

PDF 2 → Upload
       ↓
Remove PDF 1 chunks
       ↓
Index PDF 2
       ↓
Ask Questions
       ↓
Answer only from PDF 2
```

## 🧪 Testing

The application can be tested by uploading multiple PDFs one after another and asking document-specific questions.

A successful test should confirm that:

* The new PDF is indexed correctly.
* Old document chunks are removed.
* Retrieved chunks belong to the active PDF.
* Answers are based on the current document.
* Previous PDF information does not appear in the answer.

## 👨‍💻 Author

Ramesh Guddala


