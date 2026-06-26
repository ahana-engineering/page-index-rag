import re
import logging
from pypdf import PdfReader
from database import get_connection
from rank_bm25 import BM25Okapi
from langchain_core.prompts import PromptTemplate
from vector_store import store_chunk, retrieve_vector, clear_vectors
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# =====================================================
# Logging
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# LLM — Groq
# =====================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# =====================================================
# BM25 State
# =====================================================

_bm25: BM25Okapi | None = None
_chunk_lookup: list = []

# =====================================================
# Stop-words
# =====================================================

STOP_WORDS = frozenset({
    "what", "is", "the", "a", "an", "of", "in", "on",
    "for", "to", "and", "with", "about", "tell", "me",
    "list", "all", "show", "give", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should",
    "this", "that", "these", "those", "it", "its",
})

# =====================================================
# Prompt Template
# =====================================================

prompt_template = PromptTemplate(
    template="""You are an expert document analyst and research assistant. Your sole task is to answer the user's question accurately using only the facts explicitly stated in the provided context.

### CRITICAL GROUND RULES:
1. **Context-Only Boundary:** Rely ONLY on the clear facts provided below. Do NOT assume, extrapolate, or use outside knowledge. 
2. **Missing Information:** If the context does not contain the answer, you must reply exactly: `Information not found in document.` Do not attempt to write an explanation or partial answer.
3. **Synthesis:** Combine information from multiple pages or chunks seamlessly when required to form a cohesive, comprehensive answer.

### RESPONSE FORMATTING RULES:
* **Source Citations:** Every claim or statement you make must mention its source page number. Format this as `(Page X)` at the end of the sentence or bullet point.
* **Lists:** If asked to list items, use a standard numbered list (e.g., 1., 2., 3.). For other general breakdowns, use clean bullet points (`*`).
* **Completeness:** Provide fully detailed, rigorous, and complete answers. Do not truncate summaries or leave out critical context.
* **Summarization Requests:** If the user asks for a summary or overview, structure your answer to:
  - Cover all major topics discussed in the chunks.
  - Highlight key analytical findings.
  - Retain specific details, data points, or names while keeping it organized.

### CONTEXT PATTERN NOTE:
The context below consists of text snippets separated by dividers. Each snippet starts with a header indicating its origin like this: `PAGE X  |  CHUNK Y  |  Score: Z.ZZZZ`. Use this `PAGE X` identifier for your citations.

---

### PROVIDED CONTEXT:
{context}

---

### USER QUESTION:
{question}

### ANSWER:""",
    input_variables=["context", "question"],
)

# =====================================================
# Chunking
# =====================================================

def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += step
    return chunks

# =====================================================
# PDF Indexing
# =====================================================

def index_pdf(pdf_path: str, filename: str) -> int:
    reader = PdfReader(pdf_path)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chunks WHERE document_name=?", (filename,))
    clear_vectors()

    total_chunks = 0

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        page_chunks = split_text(text)

        for chunk_num, chunk in enumerate(page_chunks):
            cursor.execute(
                """
                INSERT INTO chunks
                    (document_name, page_number, chunk_number, chunk_content)
                VALUES (?, ?, ?, ?)
                """,
                (filename, page_num + 1, chunk_num + 1, chunk),
            )
            chunk_id = f"{page_num + 1}_{chunk_num + 1}"
            store_chunk(chunk_id, chunk, page_num + 1, chunk_num + 1)
            total_chunks += 1

    conn.commit()
    conn.close()

    logger.info("Indexed %d chunks from '%s'", total_chunks, filename)
    build_bm25_index()
    return total_chunks

# =====================================================
# BM25 Index Builder
# =====================================================

def build_bm25_index() -> None:
    global _bm25, _chunk_lookup

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT page_number, chunk_number, chunk_content FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    _chunk_lookup = rows

    if not rows:
        _bm25 = None
        logger.warning("build_bm25_index: no chunks found in DB.")
        return

    tokenized = [_tokenize(row[2]) for row in rows]
    _bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index built with %d chunks.", len(rows))

# =====================================================
# Tokeniser
# =====================================================

def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"\b\w+\b", text.lower())
        if tok not in STOP_WORDS
    ]

# =====================================================
# RAG MODE 1 — Page Index RAG
# Scores pages as a whole, returns all chunks from top pages
# =====================================================

def retrieve_page_index(query: str, top_k: int = 25) -> list[tuple]:
    global _bm25, _chunk_lookup

    if _bm25 is None:
        build_bm25_index()
    if _bm25 is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    raw_scores = _bm25.get_scores(tokens)

    # Aggregate BM25 scores per page
    page_scores: dict[int, float] = {}
    for i, (page_no, chunk_no, content) in enumerate(_chunk_lookup):
        page_scores[page_no] = page_scores.get(page_no, 0.0) + raw_scores[i]

    # Top 5 pages by aggregated score
    top_pages = sorted(page_scores, key=lambda p: page_scores[p], reverse=True)[:5]

    # Fetch ALL chunks from those pages
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in top_pages)
    cursor.execute(
        f"""
        SELECT page_number, chunk_number, chunk_content
        FROM chunks
        WHERE page_number IN ({placeholders})
        ORDER BY page_number, chunk_number
        """,
        top_pages,
    )
    rows = cursor.fetchall()
    conn.close()

    return [(1.0, pn, cn, c) for pn, cn, c in rows][:top_k]


# =====================================================
# RAG MODE 2 — Vector RAG
# Pure semantic similarity retrieval
# =====================================================

def retrieve_vector_only(query: str, top_k: int = 15) -> list[tuple]:
    results = retrieve_vector(query, top_k)

    if "documents" not in results or not results["documents"]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]
    max_dist = max(distances) if distances and max(distances) > 0 else 1.0

    output = []
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1.0 - (dist / max_dist)
        output.append((similarity, meta["page"], meta["chunk"], doc))

    return output


# =====================================================
# RAG MODE 3 — BM25 RAG
# Pure keyword-based retrieval
# =====================================================

def retrieve_bm25_only(query: str, top_k: int = 15) -> list[tuple]:
    global _bm25, _chunk_lookup

    if _bm25 is None:
        build_bm25_index()
    if _bm25 is None:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    raw_scores = _bm25.get_scores(tokens)
    max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0

    ranked_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:top_k]

    return [
        (
            raw_scores[i] / max_score,
            _chunk_lookup[i][0],
            _chunk_lookup[i][1],
            _chunk_lookup[i][2],
        )
        for i in ranked_indices
        if raw_scores[i] > 0
    ]


# =====================================================
# RAG MODE 4 — Proxy Pointer RAG
# Summarises candidate chunks into 1-line pointers,
# then re-ranks those pointers against the query.
# Sends only the best full chunks to the LLM.
# =====================================================

def retrieve_proxy_pointer(query: str, top_k: int = 15) -> list[tuple]:
    # Step 1: Get 20 candidate chunks via BM25
    candidates = retrieve_vector_only(query, top_k=20)
    if not candidates:
        return []

    pointer_prompt = (
        "Summarise the following text in ONE sentence that captures its key topic:\n\n"
        "{text}"
    )

    # Step 2: Summarise each chunk into a proxy pointer
    proxies = []
    for score, page_no, chunk_no, content in candidates:
        try:
            summary = llm.invoke(
                pointer_prompt.format(text=content[:300])
            ).content.strip()
        except Exception:
            summary = content[:100]
        proxies.append((summary, page_no, chunk_no, content))

    # Step 3: Re-rank proxy summaries against query via BM25
    # Step 3: Re-rank proxy summaries using Vector similarity
    import chromadb
    from chromadb.utils import embedding_functions

    # Build a temporary ChromaDB collection for proxy summaries
    temp_client = chromadb.Client()
    ef = embedding_functions.DefaultEmbeddingFunction()

    temp_col = temp_client.create_collection(
        name="proxy_temp",
        embedding_function=ef
    )

    # Add proxy summaries as temporary documents
    proxy_texts = [p[0] for p in proxies]
    temp_col.add(
        documents=proxy_texts,
        ids=[f"proxy_{i}" for i in range(len(proxy_texts))]
    )

    # Query the collection with the original user query
    results = temp_col.query(
        query_texts=[query],
        n_results=min(top_k, len(proxy_texts))
    )

    # Map results back to original chunks
    returned_ids = results["ids"][0]
    distances    = results["distances"][0]
    max_dist = max(distances) if max(distances) > 0 else 1.0

    output = []
    for doc_id, dist in zip(returned_ids, distances):
        idx = int(doc_id.split("_")[1])
        similarity = 1.0 - (dist / max_dist)
        _, page_no, chunk_no, content = proxies[idx]
        output.append((similarity, page_no, chunk_no, content))

    # Clean up temporary collection
    temp_client.delete_collection("proxy_temp")

    return output


# =====================================================
# RAG MODE 5 — Hybrid RAG (BM25 + Vector + RRF)
# =====================================================

def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def retrieve_hybrid(query: str, top_k: int = 15) -> list[tuple]:
    bm25_results = retrieve_bm25_only(query, top_k=20)
    vector_results = retrieve_vector_only(query, top_k=20)

    rrf_scores: dict[tuple, float] = {}
    chunk_data: dict[tuple, tuple] = {}

    for rank, (score, page, chunk, content) in enumerate(bm25_results):
        key = (page, chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf_score(rank)
        chunk_data[key] = (page, chunk, content)

    for rank, (score, page, chunk, content) in enumerate(vector_results):
        key = (page, chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + _rrf_score(rank)
        chunk_data[key] = (page, chunk, content)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        (rrf_scores[key], chunk_data[key][0], chunk_data[key][1], chunk_data[key][2])
        for key, _ in ranked
    ]


# =====================================================
# Context Builder
# =====================================================

def build_context(results: list[tuple]) -> tuple[str, list[int], list[str]]:
    parts = []
    pages = []
    chunk_ids = []

    for score, page_no, chunk_no, text in results:
        pages.append(page_no)
        chunk_ids.append(f"{page_no}-{chunk_no}")
        parts.append(
            f"PAGE {page_no}  |  CHUNK {chunk_no}  |  Score: {score:.4f}\n\n"
            f"{text}\n\n"
            f"{'─' * 40}"
        )

    return "\n".join(parts), pages, chunk_ids


# =====================================================
# RAG mode registry (used by Streamlit UI)
# =====================================================

RAG_MODES = {
    "⚡ Hybrid RAG":        "hybrid",
    "🔍 BM25 RAG":          "bm25",
    "🧠 Vector RAG":        "vector",
    "📄 Page Index RAG":    "page_index",
    "🔗 Proxy Pointer RAG": "proxy_pointer",
}

SUMMARY_KEYWORDS = frozenset({"summary", "summarize", "summarise", "overview"})


# =====================================================
# Ask Question — routes to selected RAG mode
# =====================================================

def ask_question(
    query: str,
    mode: str = "hybrid",
) -> tuple[str, list, list, str, str]:

    query_lower = query.lower()

    if any(kw in query_lower for kw in SUMMARY_KEYWORDS):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT page_number, chunk_number, chunk_content
            FROM chunks
            ORDER BY page_number, chunk_number
            LIMIT 40
            """
        )
        rows = cursor.fetchall()
        conn.close()
        results = [(1.0, pn, cn, c) for pn, cn, c in rows]

    elif mode == "hybrid":
        results = retrieve_hybrid(query, top_k=25)
    elif mode == "bm25":
        results = retrieve_bm25_only(query, top_k=25)
    elif mode == "vector":
        results = retrieve_vector_only(query, top_k=25)
    elif mode == "page_index":
        results = retrieve_page_index(query, top_k=25)
    elif mode == "proxy_pointer":
        results = retrieve_proxy_pointer(query, top_k=15)
    else:
        results = retrieve_hybrid(query, top_k=25)

    if not results:
        return ("No relevant content found.", [], [], "", "")

    context, pages, chunk_ids = build_context(results)
    prompt = prompt_template.format(context=context, question=query)
    answer = llm.invoke(prompt).content

    return answer, pages, chunk_ids, context, prompt


# =====================================================
# Save Query
# =====================================================

def save_query(
    session_code: str,
    query: str,
    pages: list,
    chunk_ids: list,
    context: str,
    prompt: str,
    answer: str,
    mode: str = "hybrid",
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO query_logs
            (session_code, user_query, retrieved_pages, retrieved_chunks,
             context_sent, prompt_sent, llm_response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_code,
            f"[{mode.upper()}] {query}",
            ",".join(map(str, pages)),
            ",".join(chunk_ids),
            context,
            prompt,
            answer,
        ),
    )
    conn.commit()
    conn.close()


# =====================================================
# Session History
# =====================================================

def get_session_history(session_code: str) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_query, retrieved_pages, retrieved_chunks,
               prompt_sent, llm_response, created_at
        FROM query_logs
        WHERE session_code = ?
        ORDER BY id DESC
        """,
        (session_code,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows
