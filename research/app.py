"""
Streamlit Web Application for Research Paper Grounded RAG Assistant.
Features layout-aware PDF ingestion, section-aware semantic retrieval,
1-click academic research questions, and verified source citations.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()

from rag.loader import PaperLoader, ExtractedDocument
from rag.chunker import SectionAwareChunker, TextChunk
from rag.vector_store import VectorIndex, SearchResult
from rag.generator import GroundedGenerator, GenerationResult
from rag.citation import CitationParser

# Page configuration
st.set_page_config(
    page_title="ScholarRAG - Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for academic look and feel
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #C7D2FE;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .score-badge {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .metric-container {
        display: flex;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 8px 14px;
        flex: 1;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Standard Research Prompts
RESEARCH_PRESETS = [
    ("🎯 Objective", "What is the objective of the paper?"),
    ("🔬 Methodology", "What methodology was used?"),
    ("📊 Datasets", "What datasets were used?"),
    ("💡 Major Findings", "What are the major findings?"),
    ("⚠️ Limitations", "What are the limitations?"),
]

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "active_paper_name" not in st.session_state:
    st.session_state.active_paper_name = None


def load_document_bytes(file_bytes: bytes, filename: str):
    """Ingest, parse, and index uploaded document."""
    loader = PaperLoader()
    doc = loader.load_pdf(file_bytes, filename=filename)
    chunker = SectionAwareChunker(chunk_size=900, chunk_overlap=150)
    chunks = chunker.chunk_document(doc)

    index = VectorIndex(collection_name="streamlit_rag")
    index.index_chunks(chunks)

    st.session_state.extracted_doc = doc
    st.session_state.chunks = chunks
    st.session_state.vector_index = index
    st.session_state.active_paper_name = filename
    st.session_state.messages = []


def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=["Gemini", "OpenAI", "Groq", "Local (Offline Mode)"],
            index=0,
            help="Select the AI provider for answer generation. Use 'Local' for instant offline extraction without API keys.",
        )

        provider_key = provider.lower()
        if "local" in provider_key:
            provider_key = "local"

        api_key = ""
        model_name = None

        if provider_key == "gemini":
            env_key = os.getenv("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=env_key, type="password", help="Enter your Google AI Studio API key.")
            model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"])
        elif provider_key == "openai":
            env_key = os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input("OpenAI API Key", value=env_key, type="password")
            model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
        elif provider_key == "groq":
            env_key = os.getenv("GROQ_API_KEY", "")
            api_key = st.text_input("Groq API Key", value=env_key, type="password")
            model_name = st.selectbox("Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])

        top_k = st.slider("Retrieved Chunks (Top-K)", min_value=2, max_value=8, value=4, step=1)

        st.divider()
        st.subheader("📄 Upload Research Paper")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload an academic research paper PDF (e.g. ArXiv, IEEE, NeurIPS).",
        )

        # Quick button to load sample paper
        sample_paper_btn = st.button("🧪 Load Sample Paper (DeepScale)")

        if sample_paper_btn:
            sample_path = Path("sample_paper.pdf")
            if not sample_path.exists():
                from sample_paper import generate_sample_paper
                generate_sample_paper(str(sample_path))
            with open(sample_path, "rb") as f:
                with st.spinner("Parsing and indexing sample paper..."):
                    load_document_bytes(f.read(), "sample_paper.pdf")
            st.success("Sample paper loaded successfully!")
            st.rerun()

        if uploaded_file is not None and uploaded_file.name != st.session_state.active_paper_name:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                bytes_data = uploaded_file.read()
                load_document_bytes(bytes_data, uploaded_file.name)
            st.success("Paper successfully indexed!")
            st.rerun()

        # Paper Metadata Summary Card
        if st.session_state.extracted_doc:
            doc = st.session_state.extracted_doc
            st.divider()
            st.markdown("### 📊 Paper Dossier")
            st.markdown(f"**Title:** {doc.title}")
            st.markdown(f"**Pages:** {doc.total_pages}")
            st.markdown(f"**Chunks:** {len(st.session_state.chunks)}")
            st.markdown(f"**Sections Identified:** {len(doc.sections)}")

    # Main Area
    st.markdown('<div class="main-title">ScholarRAG 📚</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload research papers, explore findings, and ask questions with verifiable page citations.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.extracted_doc:
        st.info("👈 Please upload a research paper PDF in the sidebar or click **'Load Sample Paper'** to get started.")
        return

    doc: ExtractedDocument = st.session_state.extracted_doc
    index: VectorIndex = st.session_state.vector_index

    # Metric stats bar
    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-val">{doc.total_pages}</div>
                <div class="metric-label">Pages</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{len(st.session_state.chunks)}</div>
                <div class="metric-label">Chunks</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{len(doc.sections)}</div>
                <div class="metric-label">Sections</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{provider}</div>
                <div class="metric-label">Mode</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Tabs
    tab_chat, tab_structure, tab_chunks = st.tabs(["💬 Question & Answer", "📑 Document Sections", "🧩 Chunks & Vectors"])

    with tab_chat:
        st.markdown("##### ⚡ Quick Research Inquiries")
        cols = st.columns(len(RESEARCH_PRESETS))
        selected_prompt = None

        for col, (label, question_text) in zip(cols, RESEARCH_PRESETS):
            if col.button(label, use_container_width=True):
                selected_prompt = question_text

        # 1-Click Synthesis button
        if st.button("📑 Generate Full Paper Dossier (All 5 Dimensions)", type="secondary"):
            selected_prompt = (
                "Provide a comprehensive, structured synthesis of the paper covering:\n"
                "1. Objective\n2. Methodology\n3. Datasets\n4. Major Findings\n5. Limitations\n"
                "Include verifiable [Page X, Section Y] citations for each finding."
            )

        # Render message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("🔍 Verified Sources & Evidence", expanded=False):
                        for cit in msg["citations"]:
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <span class="citation-badge">Page {cit['page_number']}</span>
                                    <span class="citation-badge">{cit['section']}</span>
                                    <div style="margin-top: 6px; color: #334155; font-style: italic;">
                                        "{cit.get('excerpt_snippet', '')}"
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        # Handle user prompt or button click
        user_input = st.chat_input("Ask any question about this research paper...")
        prompt_to_run = selected_prompt or user_input

        if prompt_to_run:
            st.session_state.messages.append({"role": "user", "content": prompt_to_run})
            with st.chat_message("user"):
                st.markdown(prompt_to_run)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant passages and synthesizing grounded answer..."):
                    retrieved = index.search(prompt_to_run, top_k=top_k)
                    generator = GroundedGenerator(
                        provider=provider_key,
                        model_name=model_name,
                        api_key=api_key if api_key else None,
                    )
                    gen_result: GenerationResult = generator.generate(prompt_to_run, retrieved)

                    st.markdown(gen_result.answer)

                    citations_data = [c.to_dict() for c in gen_result.citations]

                    if gen_result.citations:
                        with st.expander("🔍 Verified Sources & Evidence", expanded=True):
                            for cit in gen_result.citations:
                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                        <span class="citation-badge">Page {cit.page_number}</span>
                                        <span class="citation-badge">{cit.section}</span>
                                        <div style="margin-top: 6px; color: #334155; font-style: italic;">
                                            "{cit.excerpt_snippet}"
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to conversation history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": gen_result.answer,
                            "citations": citations_data,
                        }
                    )

    with tab_structure:
        st.subheader("📑 Document Outline & Section Structure")
        st.write(f"**Document Title:** {doc.title}")
        st.write(f"**Source File:** `{doc.filename}`")

        for sec in doc.sections:
            with st.expander(f"📍 {sec.name} (Starts Page {sec.start_page})", expanded=False):
                st.markdown(f"**Canonical Type:** `{sec.canonical_type}`")
                preview = sec.content[:800] + ("..." if len(sec.content) > 800 else "")
                st.text(preview)

    with tab_chunks:
        st.subheader("🧩 Indexed Chunks & Vector Space")
        st.write(f"Total Chunks: **{len(st.session_state.chunks)}**")

        search_chunk = st.text_input("Filter chunks by keyword:", "")
        filtered_chunks = [
            c for c in st.session_state.chunks
            if not search_chunk or search_chunk.lower() in c.content.lower() or search_chunk.lower() in c.section.lower()
        ]

        st.caption(f"Displaying {len(filtered_chunks)} of {len(st.session_state.chunks)} chunks")
        for chunk in filtered_chunks[:25]:
            with st.expander(f"Chunk `{chunk.chunk_id}` | Page {chunk.page_number} | {chunk.section}"):
                st.markdown(f"**Characters:** {chunk.char_count} | **Section:** `{chunk.section}`")
                st.markdown(f"```\n{chunk.content}\n```")


if __name__ == "__main__":
    main()
