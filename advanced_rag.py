from pathlib import Path
import hashlib
import re

from pypdf import PdfReader

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "documents"

STORAGE_DIR = (
    BASE_DIR
    / "storage"
    / "employee_rag_final"
)

PDF_DIR.mkdir(exist_ok=True)

STORAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. EMBEDDINGS
# =========================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# =========================================================
# 3. VECTOR DATABASE
# =========================================================

vectorstore = Chroma(
    collection_name="employee_documents_final",
    embedding_function=embeddings,
    persist_directory=str(STORAGE_DIR)
)


# =========================================================
# 4. TEXT SPLITTER
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


# =========================================================
# 5. LLM
# =========================================================

llm = OllamaLLM(
    model="llama3.2:latest"
)


# =========================================================
# 6. PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a strict document question-answering assistant.

Use ONLY the provided context.

Rules:

1. Never use outside knowledge.
2. Never invent information.
3. Never guess.
4. Answer only what the question asks.
5. Do not confuse Employee Name with Manager.
6. Do not confuse Department with Position.
7. Use exact information from the provided context.
8. For count questions, count only matching records.
9. If information is not present, say exactly:

I don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# 7. NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# 8. EXTRACT EMPLOYEE RECORDS
# =========================================================

def extract_employee_records(text):

    text = normalize_text(text)

    matches = list(
        re.finditer(
            r"\bEMP\d{3}\b",
            text,
            re.IGNORECASE
        )
    )

    records = []

    for i, match in enumerate(matches):

        start = match.start()

        if i + 1 < len(matches):

            end = matches[
                i + 1
            ].start()

        else:

            end = len(text)

        record = text[
            start:end
        ].strip()

        if record:

            records.append(
                {
                    "record": record,
                    "start": start
                }
            )

    return records


# =========================================================
# 9. PAGE NUMBER
# =========================================================

def get_page_number_from_position(
    page_ranges,
    position
):

    for page_number, start, end in page_ranges:

        if start <= position < end:

            return page_number

    if page_ranges:

        return page_ranges[-1][0]

    return 1


# =========================================================
# 10. INDEX PDF
# =========================================================

def index_pdf(pdf_path):

    pdf_path = Path(pdf_path)

    reader = PdfReader(
        str(pdf_path)
    )

    full_text_parts = []

    page_ranges = []

    current_position = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = page_text.strip()

        if not page_text:
            continue

        start_position = current_position

        full_text_parts.append(
            page_text
        )

        current_position += (
            len(page_text) + 1
        )

        end_position = current_position

        page_ranges.append(
            (
                page_number,
                start_position,
                end_position
            )
        )

    full_text = "\n".join(
        full_text_parts
    )

    if not full_text.strip():

        return 0

    has_employee_data = bool(
        re.search(
            r"\bEMP\d{3}\b",
            full_text,
            re.IGNORECASE
        )
    )

    chunks = []
    metadatas = []
    ids = []

    # =====================================================
    # EMPLOYEE DATA
    # =====================================================

    if has_employee_data:

        records = extract_employee_records(
            full_text
        )

        for index, item in enumerate(records):

            chunk = item["record"]

            if not chunk:
                continue

            page_number = (
                get_page_number_from_position(
                    page_ranges,
                    item["start"]
                )
            )

            chunks.append(
                chunk
            )

            metadatas.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "document_type":
                        "employee_master_data"
                }
            )

            chunk_hash = hashlib.md5(
                chunk.encode("utf-8")
            ).hexdigest()

            ids.append(
                f"{pdf_path.name}_"
                f"employee_"
                f"{index}_"
                f"{chunk_hash}"
            )

    # =====================================================
    # GENERAL DOCUMENT
    # =====================================================

    else:

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if not page_text:
                continue

            page_text = normalize_text(
                page_text
            )

            if not page_text:
                continue

            page_chunks = (
                text_splitter.split_text(
                    page_text
                )
            )

            for chunk_number, chunk in enumerate(
                page_chunks
            ):

                chunk = chunk.strip()

                if not chunk:
                    continue

                chunks.append(
                    chunk
                )

                metadatas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "document_type":
                            "general_document"
                    }
                )

                chunk_hash = hashlib.md5(
                    chunk.encode("utf-8")
                ).hexdigest()

                ids.append(
                    f"{pdf_path.name}_"
                    f"page_"
                    f"{page_number}_"
                    f"{chunk_number}_"
                    f"{chunk_hash}"
                )

    if not chunks:

        return 0

    # =====================================================
    # DELETE OLD PDF
    # =====================================================

    try:

        existing = vectorstore.get(
            where={
                "source": pdf_path.name
            }
        )

        old_ids = existing.get(
            "ids",
            []
        )

        if old_ids:

            vectorstore.delete(
                ids=old_ids
            )

    except Exception:

        pass

    # =====================================================
    # ADD NEW PDF
    # =====================================================

    vectorstore.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)


# =========================================================
# 11. COLLECTION COUNT
# =========================================================

def get_collection_count():

    return vectorstore._collection.count()


# =========================================================
# 12. GET ALL EMPLOYEE RECORDS
# =========================================================

def get_all_employee_records():

    data = vectorstore.get(
        where={
            "document_type":
                "employee_master_data"
        },
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

    return list(
        zip(
            documents,
            metadatas
        )
    )


# =========================================================
# 13. GET EMPLOYEE BY ID
# =========================================================

def get_employee_by_id(emp_id):

    emp_id = (
        emp_id
        .strip()
        .upper()
    )

    records = (
        get_all_employee_records()
    )

    for record, metadata in records:

        match = re.search(
            r"\bEMP\d{3}\b",
            record,
            re.IGNORECASE
        )

        if not match:
            continue

        found_id = (
            match
            .group(0)
            .upper()
        )

        if found_id == emp_id:

            return {
                "content": record,
                "metadata": metadata
            }

    return None


# =========================================================
# 14. PARSE EMPLOYEE RECORD
# =========================================================

def parse_employee_record(record):

    if not record:

        return None

    record = normalize_text(
        record
    )

    # -----------------------------------------------------
    # Employee ID
    # -----------------------------------------------------

    emp_match = re.search(
        r"\b(EMP\d{3})\b",
        record,
        re.IGNORECASE
    )

    if not emp_match:

        return None

    emp_id = (
        emp_match
        .group(1)
        .upper()
    )

    remaining = record[
        emp_match.end():
    ].strip()

    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------

    email_match = re.search(
        r"[\w.+-]+@[\w.-]+\.\w+",
        remaining
    )

    if not email_match:

        return None

    email = (
        email_match
        .group(0)
    )

    before_email = (
        remaining[
            :email_match.start()
        ]
        .strip()
    )

    after_email = (
        remaining[
            email_match.end():
        ]
        .strip()
    )

    # -----------------------------------------------------
    # Phone
    # -----------------------------------------------------

    phone_match = re.search(
        r"(?:\+91[- ]?)?\d{10}",
        after_email
    )

    if not phone_match:

        return None

    phone = (
        phone_match
        .group(0)
    )

    after_phone = (
        after_email[
            phone_match.end():
        ]
        .strip()
    )

    # -----------------------------------------------------
    # Hire Date
    # -----------------------------------------------------

    date_match = re.search(
        r"\b\d{4}-\d{2}-\d{2}\b",
        after_phone
    )

    if not date_match:

        return None

    hire_date = (
        date_match
        .group(0)
    )

    after_date = (
        after_phone[
            date_match.end():
        ]
        .strip()
    )

    # -----------------------------------------------------
    # Salary
    # -----------------------------------------------------

    salary_match = re.search(
        r"\b\d{1,3}(?:,\d{3})+\b|\b\d{5,7}\b",
        after_date
    )

    if not salary_match:

        return None

    salary = (
        salary_match
        .group(0)
    )

    after_salary = (
        after_date[
            salary_match.end():
        ]
        .strip()
    )

    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    locations = [
        "Hyderabad",
        "Delhi",
        "Mumbai",
        "Bengaluru",
        "Ahmedabad",
        "Gurugram",
        "Noida",
        "Chennai",
        "Pune",
        "Kolkata"
    ]

    location = None
    location_match = None

    for item in locations:

        match = re.search(
            rf"\b{re.escape(item)}\b",
            after_salary,
            re.IGNORECASE
        )

        if match:

            location = item
            location_match = match

            break

    if not location:

        return None

    after_location = (
        after_salary[
            location_match.end():
        ]
        .strip()
    )

    # -----------------------------------------------------
    # Status + Manager
    # -----------------------------------------------------

    status_match = re.search(
        r"\b(On Leave|Active)\b",
        after_location,
        re.IGNORECASE
    )

    if status_match:

        status = (
            status_match
            .group(1)
            .title()
        )

        manager = (
            after_location[
                :status_match.start()
            ]
            .strip()
        )

    else:

        status = "Unknown"

        manager = (
            after_location
            .strip()
        )

    # -----------------------------------------------------
    # Department
    # -----------------------------------------------------

    departments = [
        "Human Resources",
        "Engineering",
        "Marketing",
        "Product",
        "Sales",
        "Operations",
        "Finance",
        "Legal",
        "IT",
        "Customer Support"
    ]

    department = None
    department_match = None

    for dept in departments:

        match = re.search(
            rf"\b{re.escape(dept)}\b",
            before_email,
            re.IGNORECASE
        )

        if match:

            department = dept
            department_match = match

            break

    if not department:

        return None

    # -----------------------------------------------------
    # Name + Position
    # -----------------------------------------------------

    name = (
        before_email[
            :department_match.start()
        ]
        .strip()
    )

    position = (
        before_email[
            department_match.end():
        ]
        .strip()
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    position = re.sub(
        r"\s+",
        " ",
        position
    ).strip()

    manager = re.sub(
        r"\s+",
        " ",
        manager
    ).strip()

    return {
        "emp_id": emp_id,
        "name": name,
        "department": department,
        "position": position,
        "email": email,
        "phone": phone,
        "hire_date": hire_date,
        "salary": salary,
        "location": location,
        "manager": manager,
        "status": status,
        "raw_record": record
    }


# =========================================================
# 15. DETECT FIELD
# =========================================================

def detect_field(question):

    q = question.lower()

    if "salary" in q:
        return "salary"

    if "manager" in q:
        return "manager"

    if "department" in q:
        return "department"

    if "position" in q:
        return "position"

    if (
        "location" in q
        or "located" in q
    ):
        return "location"

    if "status" in q:
        return "status"

    if (
        "hire date" in q
        or "joining date" in q
        or "joined" in q
    ):
        return "hire_date"

    if "email" in q:
        return "email"

    if "phone" in q:
        return "phone"

    if "name" in q:
        return "name"

    return None


# =========================================================
# 16. EXACT EMPLOYEE QUESTION
# =========================================================

def answer_employee_id_question(
    question
):

    emp_match = re.search(
        r"\bEMP\d{3}\b",
        question,
        re.IGNORECASE
    )

    if not emp_match:

        return None

    emp_id = (
        emp_match
        .group(0)
        .upper()
    )

    employee = (
        get_employee_by_id(
            emp_id
        )
    )

    if not employee:

        return None

    parsed = parse_employee_record(
        employee["content"]
    )

    if not parsed:

        return None

    field = detect_field(
        question
    )

    if field:

        value = parsed.get(
            field
        )

        if value:

            field_name = (
                field.replace(
                    "_",
                    " "
                )
            )

            return {
                "answer":
                    f"The {field_name} "
                    f"of {parsed['emp_id']} "
                    f"is {value}.",
                "sources": [
                    {
                        "file":
                            employee["metadata"].get(
                                "source",
                                "Unknown"
                            ),
                        "page":
                            employee["metadata"].get(
                                "page",
                                "Unknown"
                            )
                    }
                ]
            }

    return None


# =========================================================
# 17. EXTRACT FILTER CONDITIONS
# =========================================================

def extract_conditions(question):

    q = question.lower()

    conditions = {}

    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    locations = [
        "hyderabad",
        "delhi",
        "mumbai",
        "bengaluru",
        "ahmedabad",
        "gurugram",
        "noida",
        "chennai",
        "pune",
        "kolkata"
    ]

    for location in locations:

        if re.search(
            rf"\b{re.escape(location)}\b",
            q
        ):

            conditions[
                "location"
            ] = location

            break

    # -----------------------------------------------------
    # Department
    # -----------------------------------------------------

    if (
        "human resources" in q
        or re.search(
            r"\bhr\b",
            q
        )
    ):

        conditions[
            "department"
        ] = "human resources"

    else:

        departments = [
            "engineering",
            "marketing",
            "product",
            "sales",
            "operations",
            "finance",
            "legal",
            "it",
            "customer support"
        ]

        for department in departments:

            if department in q:

                conditions[
                    "department"
                ] = department

                break

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    if "on leave" in q:

        conditions[
            "status"
        ] = "on leave"

    elif re.search(
        r"\bactive\b",
        q
    ):

        conditions[
            "status"
        ] = "active"

    return conditions


# =========================================================
# 18. RECORD MATCH
# =========================================================

def record_matches(
    parsed,
    conditions
):

    if (
        "location" in conditions
        and parsed["location"].lower()
        != conditions["location"].lower()
    ):

        return False

    if (
        "department" in conditions
        and parsed["department"].lower()
        != conditions["department"].lower()
    ):

        return False

    if (
        "status" in conditions
        and parsed["status"].lower()
        != conditions["status"].lower()
    ):

        return False

    return True


# =========================================================
# 19. FILTER EMPLOYEES
# =========================================================

def filter_employees(question):

    conditions = (
        extract_conditions(
            question
        )
    )

    if not conditions:

        return []

    records = (
        get_all_employee_records()
    )

    matches = []

    for record, metadata in records:

        parsed = parse_employee_record(
            record
        )

        if not parsed:

            continue

        if record_matches(
            parsed,
            conditions
        ):

            matches.append(
                {
                    "parsed": parsed,
                    "metadata": metadata
                }
            )

    return matches


# =========================================================
# 20. COUNT QUESTION
# =========================================================

def is_count_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "how many",
            "number of",
            "count",
            "total number"
        ]
    )


# =========================================================
# 21. ANSWER FILTER QUESTION
# =========================================================

def answer_filter_question(
    question,
    matches
):

    sources = []

    seen = set()

    for item in matches:

        metadata = item["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key not in seen:

            sources.append(
                {
                    "file": source,
                    "page": page
                }
            )

            seen.add(key)

    count = len(matches)

    # -----------------------------------------------------
    # COUNT
    # -----------------------------------------------------

    if is_count_question(
        question
    ):

        return {
            "answer":
                f"There are {count} employees matching the provided conditions.",
            "sources": sources
        }

    # -----------------------------------------------------
    # LIST
    # -----------------------------------------------------

    lines = []

    for index, item in enumerate(
        matches,
        start=1
    ):

        parsed = item["parsed"]

        lines.append(
            f"{index}. "
            f"{parsed['name']} "
            f"({parsed['emp_id']})"
        )

    return {
        "answer":
            "The matching employees are:\n\n"
            + "\n".join(lines),
        "sources": sources
    }


# =========================================================
# 22. MAIN ASK QUESTION
# =========================================================

def ask_question(
    question,
    top_k=5
):

    question = question.strip()

    if not question:

        return {
            "answer":
                "Please enter a question.",
            "sources": []
        }

    # =====================================================
    # FIRST: EXACT EMPLOYEE ID
    # =====================================================

    exact_answer = (
        answer_employee_id_question(
            question
        )
    )

    if exact_answer:

        return exact_answer

    # =====================================================
    # SECOND: FILTER / COUNT
    # =====================================================

    conditions = (
        extract_conditions(
            question
        )
    )

    if conditions:

        matches = filter_employees(
            question
        )

        if matches:

            return answer_filter_question(
                question,
                matches
            )

        return {
            "answer":
                "I don't know based on the provided documents.",
            "sources": []
        }

    # =====================================================
    # THIRD: NORMAL RAG
    # =====================================================

    results = (
        vectorstore.similarity_search(
            question,
            k=top_k
        )
    )

    if not results:

        return {
            "answer":
                "I don't know based on the provided documents.",
            "sources": []
        }

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = llm.invoke(
        final_prompt
    )

    sources = []

    seen_sources = set()

    for document in results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key not in seen_sources:

            sources.append(
                {
                    "file": source,
                    "page": page
                }
            )

            seen_sources.add(
                key
            )

    return {
        "answer": answer,
        "sources": sources
    }