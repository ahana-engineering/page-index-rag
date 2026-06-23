import streamlit as st

from rag import (
    get_session_history
)

# =====================================================
# PAGE TITLE
# =====================================================

st.set_page_config(
    page_title="Query History",
    layout="wide"
)

st.title(
    "Query History"
)

# =====================================================
# SESSION CODE
# =====================================================

session_code = st.text_input(
    "Session Code"
)

# =====================================================
# LOAD HISTORY
# =====================================================

if st.button(
    "Load History"
):

    if not session_code:

        st.error(
            "Please enter a Session Code."
        )

        st.stop()

    history = get_session_history(
        session_code
    )

    if history:

        st.success(
            f"{len(history)} records found"
        )

        for (
            query,
            pages,
            chunks,
            prompt,
            answer,
            timestamp
        ) in history:

            with st.expander(
                f"{timestamp} | {query}"
            ):

                # -----------------------------------------
                # Retrieved Pages
                # -----------------------------------------

                st.subheader(
                    "Retrieved Pages"
                )

                st.write(
                    pages
                )

                # -----------------------------------------
                # Retrieved Chunks
                # -----------------------------------------

                st.subheader(
                    "Retrieved Chunks"
                )

                st.write(
                    chunks
                )

                # -----------------------------------------
                # Prompt
                # -----------------------------------------

                st.subheader(
                    "Prompt Sent To Llama"
                )

                st.text(
                    prompt[:10000]
                )

                # -----------------------------------------
                # Answer
                # -----------------------------------------

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )

    else:

        st.warning(
            "No history found for this session."
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Session-Based RAG Trace History"
)