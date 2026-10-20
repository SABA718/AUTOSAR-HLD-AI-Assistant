import os
from typing import Any
import pandas as pd
import streamlit as st

from backend.coordinator import PipelineCoordinator, get_coordinator
from config.settings import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


CUSTOM_CSS = """
<style>
    /* Global Engineering Aesthetics */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* Header styling */
    .workbench-header {
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 0.85rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .workbench-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .workbench-subtitle {
        font-size: 0.82rem;
        font-weight: 500;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0.2rem 0 0 0;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 0.2rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        border-radius: 2px;
    }
    .status-dot-green {
        width: 6px;
        height: 6px;
        background-color: #16a34a;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
    }

    /* Sidebar Dark / Black Engineering Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stCaption, 
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #111827 !important;
        border-color: #334155 !important;
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #f1f5f9 !important;
        font-size: 0.88rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
    }

    /* System Status Table in Dark Sidebar */
    .status-table {
        width: 100%;
        font-size: 0.78rem;
        border-collapse: collapse;
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-table td {
        padding: 0.35rem 0;
        border-bottom: 1px solid #1e293b;
        color: #94a3b8;
    }
    .status-table td.val {
        text-align: right;
        font-weight: 600;
        color: #f8fafc;
    }

    /* Card Panels */
    .engineering-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 3px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }
    .engineering-card-header {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #475569;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.4rem;
    }

    /* Section Subheadings */
    .page-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0 0 0.2rem 0;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin: 0 0 1.25rem 0;
    }

    /* Badges */
    .badge-verified {
        display: inline-block;
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
        border-radius: 2px;
    }
    .badge-warning {
        display: inline-block;
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
        border-radius: 2px;
    }
    .badge-insufficient {
        display: inline-block;
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
        border-radius: 2px;
    }

    /* Monospace metadata tags */
    .meta-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #475569;
        background-color: #f1f5f9;
        padding: 0.15rem 0.4rem;
        border-radius: 2px;
        margin-right: 0.4rem;
        border: 1px solid #e2e8f0;
    }

    /* Clean Evidence Quote Box */
    .evidence-box {
        background-color: #f8fafc;
        border-left: 3px solid #0284c7;
        padding: 0.75rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #334155;
        line-height: 1.5;
        margin-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Metric compact containers - Dark Automotive Engineering Styling */
    div[data-testid="stMetric"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 3px !important;
        padding: 0.65rem 0.85rem !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Streamlit buttons: clean engineering aesthetic */
    button[kind="primary"] {
        background-color: #0369a1 !important;
        border-color: #0284c7 !important;
        border-radius: 2px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #075985 !important;
        border-color: #0369a1 !important;
    }

    /* Inputs */
    textarea, input {
        border-radius: 2px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }
    textarea:focus, input:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 1px #0284c7 !important;
    }
</style>
"""


def init_page_config() -> None:
    """Setup page configuration and branding."""
    st.set_page_config(
        page_title="AUTOSAR HLD AI Assistant - Engineering Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_active_coordinator() -> PipelineCoordinator:
    """Get coordinator instance cached in session state."""
    if "coordinator" not in st.session_state:
        st.session_state["coordinator"] = get_coordinator()
    return st.session_state["coordinator"]


def render_header(provider_name: str, doc_count: int) -> None:
    """Render enterprise automotive engineering header bar."""
    st.markdown(
        f"""
        <div class="workbench-header">
            <div>
                <h1 class="workbench-title">AUTOSAR HLD AI Assistant</h1>
                <p class="workbench-subtitle">Automotive Engineering Intelligence Workbench</p>
            </div>
            <div style="text-align: right; display: flex; align-items: center; gap: 0.75rem;">
                <span class="meta-tag">PROVIDER: {provider_name.upper()}</span>
                <span class="meta-tag">INDEXED: {doc_count} CHUNKS</span>
                <span class="status-badge-online">
                    <span class="status-dot-green"></span>SYSTEM ONLINE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(coordinator: PipelineCoordinator) -> str:
    """Render sidebar navigation and global configuration."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 0.6rem; border-bottom: 1px solid #1e293b; margin-bottom: 0.8rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">AUTOSAR WORKBENCH</div>
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;">HLD Specification Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("NAVIGATION")
        view_choice = st.radio(
            "Navigation",
            options=["Document Upload", "Engineering Q&A", "Architecture Explorer"],
            index=1,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
        st.caption("LLM INFERENCE SETTINGS")

        settings = get_settings()
        available_providers = ["mock"]
        if settings.gemini_api_key or os.getenv("GEMINI_API_KEY"):
            if "gemini" not in available_providers:
                available_providers.append("gemini")

        configured_default = (settings.default_llm_provider or "mock").lower()
        default_index = (
            available_providers.index(configured_default)
            if configured_default in available_providers
            else 0
        )

        provider_name = st.selectbox(
            "LLM Provider",
            options=available_providers,
            index=default_index,
            help="Select 'mock' for offline deterministic testing, or 'gemini' for live API reasoning.",
        )
        st.session_state["selected_provider"] = provider_name

        top_k = st.slider(
            "Context Depth (top_k chunks)",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of context chunks to fetch from ChromaDB vector store.",
        )
        st.session_state["top_k"] = top_k

        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
        st.caption("SYSTEM STATUS")

        total_chunks = coordinator.vector_store.count()
        active_doc = st.session_state.get("active_document_id", "None")

        status_html = f"""
        <table class="status-table">
            <tr><td>Core Service API</td><td class="val" style="color: #16a34a;">HEALTHY</td></tr>
            <tr><td>Vector DB (Chroma)</td><td class="val">{total_chunks} Chunks</td></tr>
            <tr><td>Active LLM</td><td class="val">{provider_name.upper()}</td></tr>
            <tr><td>Grounding Engine</td><td class="val" style="color: #0284c7;">ACTIVE</td></tr>
            <tr><td>Active Document</td><td class="val" style="max-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{active_doc}</td></tr>
        </table>
        """
        st.markdown(status_html, unsafe_allow_html=True)

    return view_choice


def render_upload_view(coordinator: PipelineCoordinator) -> None:
    """Render Document Upload & Ingestion View."""
    st.markdown("<div class='page-title'>Document Ingestion</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Upload and index automotive AUTOSAR High-Level Design (HLD) specifications with structural block extraction and entity recognition.</div>",
        unsafe_allow_html=True,
    )

    col_upload, col_meta = st.columns([3, 2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Select AUTOSAR HLD Specification (PDF)",
            type=["pdf"],
            help="Upload PDF documents adhering to automotive design formats.",
        )
    with col_meta:
        custom_doc_id = st.text_input(
            "Document Identifier (Optional)",
            value="",
            placeholder="e.g. AUTOSAR_BodyControl_HLD_v1",
            help="Leave blank to use the file name stem.",
        )

    if uploaded_file is not None:
        if st.button("Process & Index Document", type="primary"):
            with st.spinner("Extracting layout bounding blocks, chunking, embedding, and extracting AUTOSAR entities..."):
                try:
                    bytes_data = uploaded_file.read()
                    resolved_id = custom_doc_id.strip() if custom_doc_id.strip() else None
                    result = coordinator.process_pdf(
                        source=bytes_data,
                        filename=uploaded_file.name,
                        document_id=resolved_id,
                    )

                    st.session_state["active_document_id"] = result["document_id"]

                    st.markdown(
                        f"""
                        <div style="background-color: #0b1523; border: 1px solid #166534; border-left: 4px solid #22c55e; padding: 0.75rem 1rem; border-radius: 2px; margin: 1rem 0 1.25rem 0; font-size: 0.88rem; color: #f0fdf4; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <span style="font-weight: 700; color: #4ade80; letter-spacing: 0.03em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-right: 0.5rem;">[INGESTION SUCCESS]</span>
                                Indexed document: <code style="background-color: #1e293b; color: #38bdf8; padding: 0.15rem 0.4rem; border-radius: 2px; border: 1px solid #334155;">{result['document_id']}</code>
                                <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.4rem;">({result['filename']})</span>
                            </div>
                            <span style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #86efac; background: #064e3b; padding: 0.2rem 0.5rem; border-radius: 2px;">READY</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.caption("INGESTION METRICS SUMMARY")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Pages", result["total_pages"])
                    with col2:
                        st.metric("Indexed Chunks", result["total_chunks"])
                    with col3:
                        st.metric("SWCs Identified", result["extracted_swcs"])
                    with col4:
                        st.metric("Ports Found", result["extracted_ports"])
                    with col5:
                        st.metric("Interfaces", result["extracted_interfaces"])

                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
                    logger.error("Error during UI upload: %s", exc)


def render_qa_view(coordinator: PipelineCoordinator) -> None:
    """Render Engineering Q&A View."""
    st.markdown("<div class='page-title'>Engineering Q&A</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Query the AUTOSAR design using grounded document evidence. All claims require strict citation verification.</div>",
        unsafe_allow_html=True,
    )

    col_q, col_filter = st.columns([4, 1])
    with col_q:
        question = st.text_area(
            "Engineering Query",
            height=85,
            placeholder="Ask about software components, ports, interfaces, ECU mapping, or architecture (e.g. Which components provide vehicle speed interfaces, and what are their port directions?)...",
            label_visibility="collapsed",
        )
    with col_filter:
        doc_filter = st.text_input(
            "Filter Document ID",
            value=st.session_state.get("active_document_id", ""),
            placeholder="All documents",
            help="Filter context chunks to a specific document.",
        )

    submit_clicked = st.button("Submit Question", type="primary", disabled=not bool(question.strip()))

    if submit_clicked:
        with st.spinner("Executing dense retrieval, prompt synthesis, and citation verification..."):
            try:
                provider_name = st.session_state.get("selected_provider", "mock")
                top_k = st.session_state.get("top_k", 5)
                grounded_answer, provider, latency = coordinator.answer_query(
                    question=question.strip(),
                    document_id=doc_filter.strip() if doc_filter.strip() else None,
                    top_k=top_k,
                    provider_name=provider_name,
                )

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

                # Status and Metadata Strip
                col_badge, col_info = st.columns([1, 2])
                with col_badge:
                    if grounded_answer.has_insufficient_evidence:
                        st.markdown("<span class='badge-insufficient'>INSUFFICIENT EVIDENCE</span>", unsafe_allow_html=True)
                    elif grounded_answer.is_grounded:
                        st.markdown("<span class='badge-verified'>CITATION VERIFIED</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-warning'>GROUNDING WARNING</span>", unsafe_allow_html=True)

                with col_info:
                    st.markdown(
                        f"""
                        <div style="text-align: right;">
                            <span class="meta-tag">PROVIDER: {provider.provider_name.upper()}</span>
                            <span class="meta-tag">MODEL: {provider.model_name}</span>
                            <span class="meta-tag">LATENCY: {latency:.2f}s</span>
                            <span class="meta-tag">CITATIONS: {len(grounded_answer.verified_citations)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Report-style Answer Card
                st.markdown(
                    f"""
                    <div class="engineering-card" style="margin-top: 0.75rem;">
                        <div class="engineering-card-header">Engineering Synthesis Report</div>
                        <div style="font-size: 0.92rem; line-height: 1.6; color: #0f172a; white-space: pre-wrap;">{grounded_answer.answer_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Grounding Warnings (if any)
                if grounded_answer.grounding_warnings:
                    with st.expander("Grounding Discrepancies & Warnings", expanded=True):
                        for warn in grounded_answer.grounding_warnings:
                            st.markdown(f"- `{warn}`")

                # Verified Citations / Source Evidence Section
                if grounded_answer.verified_citations:
                    with st.expander("SOURCE EVIDENCE & CITATION AUDIT TRAIL", expanded=True):
                        for idx, cit in enumerate(grounded_answer.verified_citations, 1):
                            table_tag = " | TABLE DATA" if cit.is_table else ""
                            st.markdown(
                                f"""
                                <div style="margin-bottom: 0.85rem; padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0;">
                                    <div style="font-size: 0.82rem; font-weight: 600; color: #0369a1; font-family: 'JetBrains Mono', monospace;">
                                        [REF {idx}] Page {cit.page_number} &mdash; Section: {cit.section_title or 'General'}{table_tag}
                                    </div>
                                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 0.15rem; font-family: 'JetBrains Mono', monospace;">
                                        Document: {cit.document_id} | Chunk ID: {cit.chunk_id}
                                    </div>
                                    <div class="evidence-box">
                                        {cit.snippet}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            except Exception as exc:
                st.error(f"Query execution failed: {exc}")
                logger.error("Error during UI Q&A: %s", exc)


def render_explorer_view(coordinator: PipelineCoordinator) -> None:
    """Render Architecture Explorer View."""
    st.markdown("<div class='page-title'>Architecture Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='page-subtitle'>Inspect extracted AUTOSAR Software Components, Ports, and Interfaces structured as engineering data.</div>",
        unsafe_allow_html=True,
    )

    doc_id = st.text_input(
        "Document Identifier",
        value=st.session_state.get("active_document_id", ""),
        placeholder="Enter document identifier to query schema (e.g. AUTOSAR_BodyControl_HLD_v1)",
        help="Target document ID to explore extracted architecture entities.",
    )

    if not doc_id.strip():
        st.info("Enter an indexed Document ID or upload an AUTOSAR specification to explore architectural entities.")
        return

    entities = coordinator.get_entities(document_id=doc_id.strip())
    swcs = entities.get("software_components", [])
    ports = entities.get("ports", [])
    interfaces = entities.get("interfaces", [])
    total_count = entities.get("total_entities", 0)

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Entities", total_count)
    with m_col2:
        st.metric("Components", len(swcs))
    with m_col3:
        st.metric("Ports", len(ports))
    with m_col4:
        st.metric("Interfaces", len(interfaces))

    tab_swc, tab_port, tab_iface, tab_hier = st.tabs([
        "Software Components",
        "Ports",
        "Interfaces",
        "Hierarchical View",
    ])

    with tab_swc:
        if swcs:
            df_swcs = pd.DataFrame(swcs)
            cols_to_show = [c for c in ["name", "component_type", "description", "page_number", "section"] if c in df_swcs.columns]
            st.dataframe(df_swcs[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No Software Components identified for this document.")

    with tab_port:
        if ports:
            df_ports = pd.DataFrame(ports)
            cols_to_show = [c for c in ["name", "direction", "swc_parent", "interface_name", "description", "page_number"] if c in df_ports.columns]
            st.dataframe(df_ports[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No Ports identified for this document.")

    with tab_iface:
        if interfaces:
            df_iface = pd.DataFrame(interfaces)
            cols_to_show = [c for c in ["name", "interface_type", "description", "page_number", "section"] if c in df_iface.columns]
            st.dataframe(df_iface[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No Interfaces identified for this document.")

    with tab_hier:
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; margin-bottom: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">
                AUTOSAR Composition: Components &rarr; Associated Ports &rarr; Interfaces
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not swcs:
            st.info("No components available to build hierarchy.")
        else:
            for s in swcs:
                s_name = s.get("name", "UnknownSWC")
                s_type = s.get("component_type", "Application")
                parent_ports = [p for p in ports if (p.get("swc_parent") or "").lower() == s_name.lower()]

                with st.expander(f"SWC: {s_name} [{s_type}] &mdash; {len(parent_ports)} Port(s)", expanded=False):
                    st.caption(f"Description: {s.get('description') or 'N/A'}")
                    st.caption(f"Source: Page {s.get('page_number')} | Section: {s.get('section') or 'General'}")

                    if parent_ports:
                        for p in parent_ports:
                            p_name = p.get("name", "UnknownPort")
                            p_dir = p.get("direction", "UNKNOWN")
                            p_if = p.get("interface_name") or "None"
                            st.markdown(
                                f"""
                                <div style="margin-left: 1rem; padding: 0.35rem 0.6rem; margin-top: 0.3rem; background-color: #f8fafc; border-left: 2px solid #0284c7; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                                    &bull; <strong>{p_name}</strong> [{p_dir}] &rarr; Interface: <code>{p_if}</code>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No child ports explicitly bound to this component.")


def main() -> None:
    """Main Streamlit application entry point."""
    init_page_config()
    coord = get_active_coordinator()

    # Determine global header metadata
    provider_name = st.session_state.get("selected_provider", "gemini" if get_settings().gemini_api_key else "mock")
    doc_count = coord.vector_store.count()
    render_header(provider_name=provider_name, doc_count=doc_count)

    view = render_sidebar(coord)

    if view == "Document Upload":
        render_upload_view(coord)
    elif view == "Engineering Q&A":
        render_qa_view(coord)
    elif view == "Architecture Explorer":
        render_explorer_view(coord)


if __name__ == "__main__":
    main()
