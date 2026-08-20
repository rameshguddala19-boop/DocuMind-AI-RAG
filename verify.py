import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = BASE_DIR / "storage"
DOCUMENTS_DIR = STORAGE_DIR / "documents"
CHROMA_DIR = STORAGE_DIR / "chroma"


def check_environment():

    print("\n==============================")
    print(" ADVANCED RAG VERIFICATION")
    print("==============================\n")

    # -------------------------
    # Python
    # -------------------------

    import sys

    print(
        f"Python version: "
        f"{sys.version.split()[0]}"
    )

    # -------------------------
    # OpenAI Key
    # -------------------------

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if api_key:

        print(
            "✅ OPENAI_API_KEY found"
        )

    else:

        print(
            "❌ OPENAI_API_KEY missing"
        )

    # -------------------------
    # Directories
    # -------------------------

    DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"✅ Documents directory: "
        f"{DOCUMENTS_DIR}"
    )

    print(
        f"✅ Chroma directory: "
        f"{CHROMA_DIR}"
    )

    # -------------------------
    # Packages
    # -------------------------

    packages = [
        "chromadb",
        "openai",
        "pypdf",
        "streamlit",
        "dotenv"
    ]

    print("\nPackage check:")

    for package in packages:

        try:

            __import__(package)

            print(
                f"✅ {package}"
            )

        except ImportError:

            print(
                f"❌ {package} not installed"
            )

    # -------------------------
    # PDFs
    # -------------------------

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    print(
        f"\nPDF files found: "
        f"{len(pdf_files)}"
    )

    for pdf in pdf_files:

        print(
            f"  📄 {pdf.name}"
        )

    print(
        "\n=============================="
    )

    print(
        "Verification completed."
    )

    print(
        "==============================\n"
    )


if __name__ == "__main__":
    check_environment()