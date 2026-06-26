import os
import uuid
import streamlit as st

from rag import (
    index_pdf,
    ask_question,
    save_query,
    build_bm25_index,
    RAG_MODES,
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="RAG Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
/* Mode cards */
div[data-testid="column"] > div {
    border-radius: 10px;
}
/* Active mode badge */
.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    background: #1c6ef3;
    color: white;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION
# =====================================================

if "session_code" not in st.session_state:
    st.session_state.session_code = str(uuid.uuid4())[:8]

if "bm25_ready" not in st.session_state:
    build_bm25_index()
    st.session_state.bm25_ready = True

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None

session_code = st.session_state.session_code

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("### 🗂 Session")
    st.success(f"ID: `{session_code}`")
    st.divider()

    st.markdown("### 📄 Active Document")
    if "indexed_file" in st.session_state:
        st.info(st.session_state.indexed_file)
    else:
        st.caption("No document indexed yet.")

    st.divider()

    if st.session_state.selected_mode:
        st.markdown("### ⚙️ Active RAG Mode")
        st.markdown(
            f'<div class="mode-badge">{st.session_state.selected_mode}</div>',
            unsafe_allow_html=True
        )
        label_to_desc = {
            "⚡ Hybrid RAG":        "BM25 + Vector + RRF fusion",
            "🔍 BM25 RAG":          "Keyword-based retrieval",
            "🧠 Vector RAG":        "Semantic similarity search",
            "📄 Page Index RAG":    "Full-page context retrieval",
            "🔗 Proxy Pointer RAG": "Summary-guided retrieval",
        }
        desc = label_to_desc.get(st.session_state.selected_mode, "")
        st.caption(desc)

        if st.button("↩ Change Mode"):
            st.session_state.selected_mode = None
            st.rerun()

    st.divider()
    st.caption("Powered by Groq · Llama 3.3 70B")

# =====================================================
# LANDING — MODE SELECTOR
# =====================================================

if st.session_state.selected_mode is None:

    st.title("📚 RAG Playground")
    st.markdown("##### Choose a retrieval strategy to get started")
    st.divider()

    mode_info = {
        "⚡ Hybrid RAG": {
            "desc": "Combines BM25 keywords + vector semantics using Reciprocal Rank Fusion. Best overall retrieval quality.",
            "best": "General purpose — recommended for most documents",
            "speed": "⚡ Fast",
        },
        "🔍 BM25 RAG": {
            "desc": "Pure keyword matching. Works best when your question uses the same words as the document.",
            "best": "Exact terms, names, codes, IDs",
            "speed": "⚡ Fastest",
        },
        "🧠 Vector RAG": {
            "desc": "Finds chunks by meaning, not keywords. Works even when wording differs from the document.",
            "best": "Conceptual questions, paraphrased queries",
            "speed": "🟡 Moderate",
        },
        "📄 Page Index RAG": {
            "desc": "Scores entire pages and returns all chunks from the top pages. Avoids chunk boundary cuts.",
            "best": "Questions needing full page context, lists, tables",
            "speed": "⚡ Fast",
        },
        "🔗 Proxy Pointer RAG": {
            "desc": "Summarises candidate chunks into pointers, re-ranks them, then retrieves the best full chunks.",
            "best": "Complex multi-hop questions, noisy documents",
            "speed": "🔴 Slower (extra LLM calls)",
        },
    }

    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2, col1]

    for i, (label, info) in enumerate(mode_info.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"### {label}")
                st.caption(info["desc"])
                st.markdown(f"**Best for:** {info['best']}")
                st.markdown(f"**Speed:** {info['speed']}")
                if st.button(f"Select", key=f"select_{i}"):
                    st.session_state.selected_mode = label
                    st.rerun()

    st.stop()

# =====================================================
# MAIN APP — after mode is selected
# =====================================================

selected_label = st.session_state.selected_mode
selected_mode_key = RAG_MODES[selected_label]

st.title(f"📚 RAG Playground  ·  {selected_label}")
st.caption(f"Session: `{session_code}`  ·  LLM: Llama 3.3 70B via Groq")
st.divider()

# =====================================================
# STEP 1 — PDF UPLOAD
# =====================================================

st.subheader("1 · Upload & Index a PDF")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    pdf_path = os.path.join("uploads", uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("⚙️ Create Index"):
        with st.spinner("Chunking and indexing — this may take a moment…"):
            try:
                total = index_pdf(pdf_path, uploaded_file.name)
                st.session_state.indexed_file = uploaded_file.name
                st.success(f"✅ Indexed **{total}** chunks from *{uploaded_file.name}*")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

if "indexed_file" in st.session_state:
    st.info(f"Active document: **{st.session_state.indexed_file}**")

# =====================================================
# STEP 2 — QUESTION
# =====================================================

st.divider()
st.subheader("2 · Ask a Question")

query = st.text_input(
    "Enter your question",
    placeholder="e.g. What are the main findings?"
)

if st.button("🔍 Search"):

    if "indexed_file" not in st.session_state:
        st.warning("Please upload and index a PDF first.")
        st.stop()

    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner(f"Searching with {selected_label}…"):
        try:
            answer, pages, chunk_ids, context, prompt = ask_question(
                query,
                mode=selected_mode_key
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    try:
        save_query(
            session_code, query, pages, chunk_ids,
            context, prompt, answer, mode=selected_mode_key
        )
    except Exception:
        pass

    # -------------------------------------------------
    # RESULTS
    # -------------------------------------------------

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📑 Retrieved Pages")
        unique_pages = sorted(set(pages))
        st.write(", ".join(f"Page {p}" for p in unique_pages) if unique_pages else "—")

        st.subheader("🧩 Chunk IDs")
        st.write(", ".join(chunk_ids) if chunk_ids else "—")

        st.subheader("⚙️ Mode Used")
        st.markdown(
            f'<div class="mode-badge">{selected_label}</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("💬 Answer")
        st.write(answer)

    with st.expander("🔧 Prompt sent to LLM"):
        st.text(prompt[:10_000])

    with st.expander("📄 Retrieved Context"):
        st.text(context[:10_000])

# =====================================================
# FOOTER
# =====================================================

st.divider()
st.caption(
    "RAG Playground · BM25 · ChromaDB · Page Index · Proxy Pointer · Hybrid RRF · "
    "Groq · Llama 3.3 70B · SQLite"
)
