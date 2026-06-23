from pypdf import PdfReader

from database import get_connection

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


# =====================================================
# Llama Model
# =====================================================

llm = OllamaLLM(
    model="llama3.2"
)

# =====================================================
# Prompt Template
# =====================================================

prompt_template = PromptTemplate(
    template="""
You are a document assistant.

Answer ONLY from the supplied context.

Context:
{context}

Question:
{question}

If the answer is not found in the context,
reply exactly:

Information not found in document.
""",
    input_variables=[
        "context",
        "question"
    ]
)

# =====================================================
# Chunking
# =====================================================

def split_text(
        text,
        chunk_size=800,
        overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += (
            chunk_size - overlap
        )

    return chunks

# =====================================================
# PDF Indexing
# =====================================================

def index_pdf(
        pdf_path,
        filename):

    reader = PdfReader(
        pdf_path
    )

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM chunks
        WHERE document_name=?
        """,
        (filename,)
    )

    for page_num, page in enumerate(
            reader.pages):

        text = page.extract_text()

        if not text:
            continue

        page_chunks = split_text(
            text
        )

        for chunk_num, chunk in enumerate(
                page_chunks):

            cursor.execute(
                """
                INSERT INTO chunks
                (
                    document_name,
                    page_number,
                    chunk_number,
                    chunk_content
                )
                VALUES
                (?,?,?,?)
                """,
                (
                    filename,
                    page_num + 1,
                    chunk_num + 1,
                    chunk
                )
            )

    conn.commit()
    conn.close()

# =====================================================
# Retrieval
# =====================================================

def retrieve_chunks(
        query,
        top_k=20):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            page_number,
            chunk_number,
            chunk_content
        FROM chunks
        """
    )

    rows = cursor.fetchall()

    conn.close()

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "with",
        "about",
        "tell",
        "me"
    }

    query_words = [

        word.lower()

        for word in query.split()

        if word.lower()
        not in stop_words
    ]

    scores = []

    for (
        page_no,
        chunk_no,
        content
    ) in rows:

        score = 0

        content_lower = (
            content.lower()
        )

        for word in query_words:

            score += content_lower.count(
                word
            )

        scores.append(
            (
                score,
                page_no,
                chunk_no,
                content
            )
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:top_k]

# =====================================================
# Context Builder
# =====================================================

def build_context(
        results):

    context = ""

    pages = []

    chunk_ids = []

    for (
        score,
        page_no,
        chunk_no,
        text
    ) in results:

        pages.append(
            page_no
        )

        chunk_ids.append(
            f"{page_no}-{chunk_no}"
        )

        context += f"""

PAGE {page_no}
CHUNK {chunk_no}

{text}

-------------------------------------

"""

    return (
        context,
        pages,
        chunk_ids
    )

# =====================================================
# Ask Question
# =====================================================

def ask_question(
        query):

    query_lower = (
        query.lower()
    )

    # ----------------------------------
    # Summary Mode
    # ----------------------------------

    if (
        "summary" in query_lower
        or "summarize" in query_lower
        or "overview" in query_lower
    ):

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                page_number,
                chunk_number,
                chunk_content
            FROM chunks
            ORDER BY
                page_number,
                chunk_number
            LIMIT 20
            """
        )

        rows = cursor.fetchall()

        conn.close()

        results = []

        for (
            page_no,
            chunk_no,
            content
        ) in rows:

            results.append(
                (
                    1,
                    page_no,
                    chunk_no,
                    content
                )
            )

    else:

        results = retrieve_chunks(
            query
        )

    if not results:

        return (
            "No relevant content found.",
            [],
            [],
            "",
            ""
        )

    context, pages, chunk_ids = (
        build_context(
            results
        )
    )

    prompt = (
        prompt_template.format(
            context=context,
            question=query
        )
    )

    answer = llm.invoke(
        prompt
    )

    return (
        answer,
        pages,
        chunk_ids,
        context,
        prompt
    )

# =====================================================
# Save Query
# =====================================================

def save_query(
        session_code,
        query,
        pages,
        chunk_ids,
        context,
        prompt,
        answer):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO query_logs
        (
            session_code,
            user_query,
            retrieved_pages,
            retrieved_chunks,
            context_sent,
            prompt_sent,
            llm_response
        )
        VALUES
        (?,?,?,?,?,?,?)
        """,
        (
            session_code,
            query,
            ",".join(
                map(
                    str,
                    pages
                )
            ),
            ",".join(
                chunk_ids
            ),
            context,
            prompt,
            answer
        )
    )

    conn.commit()

    conn.close()

# =====================================================
# History
# =====================================================

def get_session_history(
        session_code):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT

            user_query,

            retrieved_pages,

            retrieved_chunks,

            prompt_sent,

            llm_response,

            created_at

        FROM query_logs

        WHERE session_code=?

        ORDER BY id DESC
        """,
        (session_code,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows