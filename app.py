import streamlit as st
from dotenv import load_dotenv

from advanced_rag import (
    PDF_DIR,
    index_pdf,
    ask_question,
    get_collection_count,
    vectorstore
)


load_dotenv()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("📚 DocuMind AI")

st.write(
    "Upload your documents and ask questions "
    "using Retrieval-Augmented Generation."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload PDF file",
        type=["pdf"],
        accept_multiple_files=False
    )

    if uploaded_file:

        st.write(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Process Document",
            use_container_width=True
        ):

            file_path = (
                PDF_DIR /
                uploaded_file.name
            )

            try:

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )

                chunks = index_pdf(
                    file_path
                )

                st.success(
                    f"{uploaded_file.name}: "
                    f"{chunks} chunks indexed"
                )

            except Exception as e:

                st.error(
                    f"{uploaded_file.name}: "
                    f"{str(e)}"
                )

    st.divider()


    # =====================================================
    # VECTOR DATABASE
    # =====================================================

    st.subheader("📊 Vector Database")

    st.metric(
        "Indexed Chunks",
        get_collection_count()
    )


    # =====================================================
    # VIEW CHUNKS
    # =====================================================

    if st.button(
        "🔍 View Indexed Chunks",
        use_container_width=True
    ):

        st.subheader("🧩 Indexed Chunks")

        try:

            data = vectorstore.get(
                include=[
                    "documents",
                    "metadatas"
                ]
            )

            documents = data.get(
                "documents",
                []
            ) or []

            metadatas = data.get(
                "metadatas",
                []
            ) or []

            if not documents:

                st.info(
                    "No chunks found. "
                    "Please process a PDF first."
                )

            else:

                st.write(
                    f"Total chunks: "
                    f"**{len(documents)}**"
                )

                for index, document in enumerate(
                    documents,
                    start=1
                ):

                    with st.expander(
                        f"Chunk {index}"
                    ):

                        st.write(
                            document
                        )

                        if (
                            index - 1
                            <
                            len(metadatas)
                        ):

                            metadata = (
                                metadatas[
                                    index - 1
                                ]
                            )

                            st.caption(
                                f"📄 Source: "
                                f"{metadata.get('source', 'Unknown')} "
                                f"| Page: "
                                f"{metadata.get('page', 'Unknown')}"
                            )

        except Exception as e:

            st.error(
                f"Could not load chunks: {str(e)}"
            )


# =========================================================
# QUESTION AREA
# =========================================================

st.subheader("💬 Ask Your Documents")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What is the salary of EMP049?"
    ),
    height=120
)


# =========================================================
# TOP K
# =========================================================

top_k = st.slider(
    "Number of documents to retrieve",
    min_value=1,
    max_value=10,
    value=5
)


# =========================================================
# ASK BUTTON
# =========================================================

if st.button(
    "🔎 Ask Question",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif get_collection_count() == 0:

        st.warning(
            "Please upload and process a PDF document first."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = ask_question(
                    question,
                    top_k=top_k
                )


                # =========================================
                # ANSWER
                # =========================================

                st.subheader("🤖 Answer")

                st.write(
                    result["answer"]
                )


                # =========================================
                # SOURCES
                # =========================================

                st.divider()

                st.subheader("📌 Sources")

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    for source in sources:

                        st.write(
                            f"📄 **{source['file']}** "
                            f"| Page: **{source['page']}**"
                        )

                else:

                    st.write(
                        "No source information available."
                    )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DocuMind AI • Advanced RAG Document Question Answering"
)