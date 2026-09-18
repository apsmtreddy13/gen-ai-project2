"""
Generates a formal 2-page PROJECT MANUAL PDF matching the exact structure,
fonts, table formatting, and layout of the reference project manual.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically add footer to each page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#333333"))
        footer_text = "Project Manual - Research Paper Retrieval-Augmented Generation (RAG) System"
        self.drawString(54, 36, footer_text)
        self.restoreState()


def build_pdf(filename: str = "PROJECT_MANUAL.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=40,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,  # Center
        textColor=colors.black,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        alignment=1,  # Center
        textColor=colors.black,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=5,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A1A1A"),
        leftIndent=14,
        spaceAfter=2.5,
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
    )

    story = []

    # ==================== PAGE 1 ====================
    story.append(Paragraph("PROJECT MANUAL", title_style))
    story.append(
        Paragraph(
            "RESEARCH PAPER RETRIEVAL-AUGMENTED GENERATION<br/>(RAG) SYSTEM",
            subtitle_style,
        )
    )

    # Metadata Table
    table_data = [
        [
            Paragraph("Student Name", table_header_style),
            Paragraph("Jagan", table_cell_style),
        ],
        [
            Paragraph("Registration Number", table_header_style),
            Paragraph("231FA23079", table_cell_style),
        ],
        [
            Paragraph("Email", table_header_style),
            Paragraph("jagangannavarapu2004@gmail.com", table_cell_style),
        ],
        [
            Paragraph("GitHub Repository", table_header_style),
            Paragraph("https://github.com/Jagan240804/Research-paper-question-answering-system", table_cell_style),
        ],
    ]

    col_widths = [160, 344]
    meta_table = Table(table_data, colWidths=col_widths)
    meta_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8F9FA")),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 1. Project Overview
    story.append(Paragraph("1. Project Overview", heading_style))
    story.append(
        Paragraph(
            "The Research Paper RAG System is an intelligent document-based question-answering application "
            "engineered to analyze academic literature and synthesize context-grounded answers with verifiable "
            "page-level source citations. Instead of generic language model generation that suffers from confabulation "
            "and hallucination, the system reconstructs complex multi-column scientific layouts, performs section-bounded "
            "semantic chunking, executes dense vector retrieval, and enforces strict factual provenance for all generated claims.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The project pairs a high-performance Python backend with an interactive Streamlit web interface and a CLI. "
            "This modular separation provides a structured pipeline for layout-aware document ingestion, ChromaDB vector "
            "indexing, multi-provider LLM synthesis (Gemini, OpenAI, Groq), and local extractive offline processing.",
            body_style,
        )
    )

    # 2. Project Objective
    story.append(Paragraph("2. Project Objective", heading_style))
    story.append(
        Paragraph(
            "The primary objective is to allow users to upload research papers and extract factual, verifiable answers "
            "to essential research questions including the paper's overarching objective, methodology, benchmark datasets, "
            "major empirical findings, and stated limitations. Every synthesized answer must be explicitly grounded in the "
            "source text with precise <b>[Page X, Section Y]</b> citations.",
            body_style,
        )
    )

    # 3. How the System Works
    story.append(Paragraph("3. How the System Works", heading_style))
    steps = [
        "<b>1. PDF Ingestion:</b> The user uploads an academic research paper in PDF format.",
        "<b>2. Layout-Aware Parsing:</b> PyMuPDF extracts text blocks and coordinates, sorting columns to prevent reading crossover.",
        "<b>3. Section Detection:</b> Academic section headers (Abstract, Introduction, Methodology, etc.) are detected and cataloged.",
        "<b>4. Section-Bounded Chunking:</b> Text is split into 800-1,000 character windows preserving section context and page numbers.",
        "<b>5. Dense Vector Embedding:</b> BAAI/bge-small-en-v1.5 embeddings are generated via FastEmbed ONNX and stored in ChromaDB.",
        "<b>6. Semantic Retrieval & Boosting:</b> User queries retrieve top-k chunks with priority boosts for target academic sections.",
        "<b>7. Grounded Generation & Citation:</b> LLM synthesizes context-confined answers with verifiable page-level citations.",
    ]
    for step in steps:
        story.append(Paragraph(step, bullet_style))

    # 4. Technologies Used
    story.append(Paragraph("4. Technologies Used", heading_style))
    story.append(
        Paragraph(
            "The project uses <b>Python 3.10</b> for backend processing and RAG orchestration, <b>Streamlit</b> for the interactive "
            "web interface, <b>PyMuPDF (fitz)</b> for layout-aware PDF parsing and column ordering, <b>ChromaDB</b> for persistent "
            "and in-memory vector storage, <b>FastEmbed (BGE-small ONNX)</b> for lightweight CPU semantic embeddings, "
            "<b>Google Gemini / OpenAI / Groq SDKs</b> for conversational grounded generation, and <b>Pytest</b> for automated test verification.",
            body_style,
        )
    )

    # 5. Retrieval Process - Simple Explanation
    story.append(Paragraph("5. Retrieval Process - Simple Explanation", heading_style))
    story.append(
        Paragraph(
            "When a user asks a question (such as <i>'What datasets were used?'</i>), the system converts the query into a high-dimensional "
            "vector and performs cosine similarity search across the paper's indexed chunks. Chunks originating from relevant sections "
            "(e.g., <i>'Experiments & Datasets'</i>) receive an algorithmic relevance boost. The top-ranked passages are combined into "
            "a provenance-labeled prompt that strictly instructs the language model to answer solely using the provided excerpts and cite "
            "the corresponding page numbers.",
            body_style,
        )
    )

    # 6. Application Workflow (Heading on Page 1, items on Page 2)
    story.append(Paragraph("6. Application Workflow", heading_style))

    # ==================== PAGE 2 ====================
    story.append(PageBreak())

    app_steps = [
        "&bull; Launch the RAG application (Streamlit web dashboard or CLI).",
        "&bull; Upload a research paper PDF or click 'Load Sample Paper (DeepScale)'.",
        "&bull; Review extracted document metrics (total pages, detected sections, total chunks).",
        "&bull; Select an LLM provider (Google Gemini, OpenAI, Groq, or Local Offline Mode).",
        "&bull; Click a 1-click research preset (Objective, Methodology, Datasets, Findings, Limitations).",
        "&bull; View grounded answer and expand 'Verified Sources & Evidence' to examine citations and excerpts.",
    ]
    for stp in app_steps:
        story.append(Paragraph(stp, bullet_style))

    # 7. Project Architecture
    story.append(Paragraph("7. Project Architecture", heading_style))
    story.append(
        Paragraph(
            "The system is built on a modular pipeline architecture with clean separation between ingestion, indexing, retrieval, and generation:<br/>"
            "&bull; <b>Loader Layer (rag/loader.py):</b> Block-coordinate PDF parser handling multi-column formats and section boundaries.<br/>"
            "&bull; <b>Chunker Layer (rag/chunker.py):</b> Semantic sliding-window chunker tagging every chunk with page and section metadata.<br/>"
            "&bull; <b>Vector Store Layer (rag/vector_store.py):</b> ChromaDB client with FastEmbed embeddings and BM25 fallback.<br/>"
            "&bull; <b>Generator Layer (rag/generator.py):</b> Grounded generation engine with strict anti-hallucination prompts.<br/>"
            "&bull; <b>Citation Layer (rag/citation.py):</b> Regex parser verifying [Page X, Section Y] tags against physical text chunks.<br/>"
            "&bull; <b>Presentation Layer (app.py & cli.py):</b> Streamlit web dashboard and terminal interface.",
            body_style,
        )
    )

    # 8. Key Advantages
    story.append(Paragraph("8. Key Advantages", heading_style))
    advantages = [
        "&bull; <b>Layout-Aware Ingestion:</b> Accurately separates columns in standard academic papers (ArXiv, IEEE, NeurIPS).",
        "&bull; <b>Zero Hallucination:</b> Strict context-grounded prompting ensures only verifiable facts are synthesized.",
        "&bull; <b>Verifiable Source Citations:</b> Every claim is attributed to a specific page number and section heading.",
        "&bull; <b>Offline Capability:</b> Operates completely offline with local ONNX embeddings and rule-based extractive synthesis.",
        "&bull; <b>1-Click Academic Inquiries:</b> Instant answers for Objective, Methodology, Datasets, Findings, and Limitations.",
        "&bull; <b>Transparent Inspection:</b> Users can view detected document outlines and inspect underlying vector chunks.",
    ]
    for adv in advantages:
        story.append(Paragraph(adv, bullet_style))

    # 9. GitHub Source Code
    story.append(Paragraph("9. GitHub Source Code", heading_style))
    story.append(
        Paragraph(
            "The complete project source code, unit test suite, and documentation are available at:<br/>"
            "<b>GitHub Repository:</b> https://github.com/Jagan240804/Research-paper-question-answering-system<br/>"
            "<b>Student Name:</b> Jagan &nbsp;|&nbsp; <b>Registration No.:</b> 231FA23079<br/>"
            "<b>Email:</b> jagangannavarapu2004@gmail.com",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {filename}")


if __name__ == "__main__":
    build_pdf("PROJECT_MANUAL.pdf")
