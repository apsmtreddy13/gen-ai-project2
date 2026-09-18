# ScholarRAG: Context-Grounded Research Paper Assistant 📚

A specialized Retrieval-Augmented Generation (RAG) system built for scientific and research papers. Ingest multi-page academic PDFs, extract structured sections across complex multi-column layouts, perform dense semantic vector retrieval, and generate factual answers strictly grounded in context with verifiable page-level source citations.

---

## 🌟 Key Features

1. **Layout-Aware PDF Ingestion (`rag/loader.py`)**:
   - Handles multi-column layouts (IEEE, ACM, ArXiv, NeurIPS) without interweaving text across columns.
   - Cleans hyphenated line breaks (`meth-\nodology` $\rightarrow$ `methodology`).
   - Automatically detects academic sections (Abstract, Introduction, Methodology, Datasets, Results, Limitations, References).

2. **Section-Aware Semantic Chunking (`rag/chunker.py`)**:
   - Bounded by semantic paragraph breaks and section headers.
   - Preserves rich metadata with every chunk: `paper_title`, `page_number`, `section`, `canonical_section`.

3. **Dense Vector Retrieval (`rag/vector_store.py`)**:
   - Fast local ONNX semantic embeddings (`BAAI/bge-small-en-v1.5`) via FastEmbed & ChromaDB.
   - Academic section re-ranking boost (e.g. boosting Methodology sections for architecture queries, Experiments for dataset queries).
   - In-memory BM25 lexical fallback for zero-dependency resilience.

4. **Context-Grounded Generation (`rag/generator.py`)**:
   - Strict anti-hallucination system prompt: answers are strictly confined to retrieved passages.
   - Multi-provider support:
     - **Google Gemini** (`gemini-1.5-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`)
     - **OpenAI** (`gpt-4o-mini`, `gpt-4o`)
     - **Groq** (`llama-3.3-70b-versatile`, `mixtral-8x7b-32768`)
     - **Local / Offline Mode**: High-precision extractive synthesizer requiring zero external API keys.

5. **Source Citation Engine (`rag/citation.py`)**:
   - Enforces claims to be cited as `[Page X, Section Y]`.
   - Links every claim to the physical page and section in the paper.
   - Provides expandable source cards with text evidence snippets.

6. **Targeted Research Analysis**:
   - 🎯 **Objective**: *"What is the objective of the paper?"*
   - 🔬 **Methodology**: *"What methodology was used?"*
   - 📊 **Datasets**: *"What datasets were used?"*
   - 💡 **Major Findings**: *"What are the major findings?"*
   - ⚠️ **Limitations**: *"What are the limitations?"*
   - 📑 **Full Paper Dossier**: Generates a unified executive summary covering all five areas in one shot.

---

## 🚀 Quick Start

### 1. Launch the Streamlit Web Application
```bash
.venv\Scripts\streamlit run app.py
```
Open your browser to `http://localhost:8501`.
- Drag and drop any research paper PDF, or click **"🧪 Load Sample Paper (DeepScale)"** to test immediately.
- Use the quick action pills to interrogate the paper's objective, methodology, datasets, findings, and limitations.
- Expand **"🔍 Verified Sources & Evidence"** under any answer to inspect cited pages and excerpts.

---

### 2. Command-Line Interface (CLI)

You can also run queries directly from your terminal:

```bash
# Generate the sample paper if not already present
.venv\Scripts\python sample_paper.py

# Query a specific research dimension
.venv\Scripts\python cli.py sample_paper.pdf --preset objective --provider local
.venv\Scripts\python cli.py sample_paper.pdf --preset methodology --provider local
.venv\Scripts\python cli.py sample_paper.pdf --preset datasets --provider local
.venv\Scripts\python cli.py sample_paper.pdf --preset findings --provider local
.venv\Scripts\python cli.py sample_paper.pdf --preset limitations --provider local

# Run all 5 dimensions sequentially
.venv\Scripts\python cli.py sample_paper.pdf --preset all --provider local

# Ask a custom question
.venv\Scripts\python cli.py sample_paper.pdf --question "What was the accuracy on PubMedQA?" --provider local
```

---

### 3. API Key Setup (Optional)
To use conversational LLMs (Gemini, OpenAI, or Groq), enter your API key in the Streamlit sidebar settings or configure `.env`:
```bash
cp .env.example .env
```
Add your key:
```env
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```
*(If no API key is provided, ScholarRAG automatically operates in local extractive mode with full source citations).*

---

### 4. Running Automated Tests
```bash
.venv\Scripts\python -m pytest tests/test_rag.py -v
```
All 6 tests verify:
- Synthetic PDF generation
- Multi-column layout extraction & section discovery
- Section-aware chunking and page metadata tagging
- Lexical and dense retrieval
- Citation extraction and chunk mapping
- End-to-end evaluation across all 5 core research questions
