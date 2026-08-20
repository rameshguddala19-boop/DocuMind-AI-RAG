import streamlit as st
from pypdf import PdfReader
import ollama
import chromadb
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=api_key)

st.title("🧠 DocuMind AI")

st.write("Ask questions. Get answers from your documents.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    pdf = PdfReader(uploaded_file)

    full_text = ""

    for page in pdf.pages:
        full_text += page.extract_text()

    chunk_size = 100
    overlap = 20

    chunks = []

    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)

        start = end - overlap

    embeddings = []

    for chunk in chunks:
        response = ollama.embed(
            model="nomic-embed-text",
            input=chunk
        )

        embeddings.append(
            response["embeddings"][0]
        )

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="rag_documents"
    )

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

    st.write("PDF uploaded successfully!")

    question = st.text_input(
        "Ask a question about your document:"
    )

    if question:

        question_response = ollama.embed(
            model="nomic-embed-text",
            input=question
        )

        question_embedding = (
            question_response["embeddings"][0]
        )

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=2
        )

        context = "\n".join(
            results["documents"][0]
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question using only the provided document context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ]
        )

        answer = response.choices[0].message.content

        st.write("### Answer")
        st.write(answer)