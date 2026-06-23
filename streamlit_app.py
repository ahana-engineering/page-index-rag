import os
import streamlit as st
import uuid

from rag import (
    index_pdf,
    ask_question,
    save_query
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Page Index RAG",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title(
    "Page Index RAG with Llama 3.2"
)

# =====================================================
# SESSION CODE
# =====================================================

st.sidebar.header(
    "Session"
)

if "session_code" not in st.session_state:

    st.session_state.session_code = (
        str(uuid.uuid4())[:8]
    )

session_code = (
    st.session_state.session_code
)

st.sidebar.success(
    f"Session: {session_code}"
)

# =====================================================
# PDF UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(
        pdf_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    if st.button(
        "Create Page Index"
    ):

        with st.spinner(
            "Creating chunks..."
        ):

            index_pdf(
                pdf_path,
                uploaded_file.name
            )

        st.success(
            "Chunk Index Created Successfully"
        )

# =====================================================
# QUERY SECTION
# =====================================================

st.subheader(
    "Ask Questions"
)

query = st.text_input(
    "Enter your question"
)

# =====================================================
# SEARCH
# =====================================================

if st.button(
    "Search"
):

    if not session_code:

        st.error(
            "Please enter a Session Code."
        )

        st.stop()

    if not query:

        st.error(
            "Please enter a question."
        )

        st.stop()

    with st.spinner(
        "Searching document..."
    ):

        (
            answer,
            pages,
            chunk_ids,
            context,
            prompt
        ) = ask_question(
            query
        )

        save_query(
            session_code,
            query,
            pages,
            chunk_ids,
            context,
            prompt,
            answer
        )

    # =================================================
    # RETRIEVED PAGES
    # =================================================

    st.subheader(
        "Retrieved Pages"
    )

    unique_pages = sorted(
        list(
            set(pages)
        )
    )

    if unique_pages:

        st.write(
            ", ".join(
                [
                    f"Page {p}"
                    for p in unique_pages
                ]
            )
        )

    # =================================================
    # RETRIEVED CHUNKS
    # =================================================

    st.subheader(
        "Retrieved Chunks"
    )

    st.write(
        chunk_ids
    )

    # =================================================
    # ANSWER
    # =================================================

    st.subheader(
        "Answer"
    )

    st.write(
        answer
    )

    # =================================================
    # PROMPT
    # =================================================

    with st.expander(
        "Prompt Sent To Llama"
    ):

        st.text(
            prompt[:10000]
        )

    # =================================================
    # CONTEXT
    # =================================================

    with st.expander(
        "Retrieved Context"
    ):

        st.text(
            context[:10000]
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Vectorless Page Index RAG | Chunking | SQLite | Llama 3.2 | Session Logging"
)