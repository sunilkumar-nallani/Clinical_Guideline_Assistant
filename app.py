import streamlit as st
import openai

# --------------------------------------------------------------------------
# Pipeline import — UI still loads and shows a clear message if index is
# missing (e.g., fresh clone before running Build_Index.py)
# --------------------------------------------------------------------------
try:
    from Rag_Pipeline import answer_question, AVAILABLE_SOURCES
    PIPELINE_READY = True
    PIPELINE_ERROR = None
except Exception as e:  # noqa: BLE001
    PIPELINE_READY = False
    PIPELINE_ERROR = str(e)
    AVAILABLE_SOURCES = []

    def answer_question(_question: str, source_type: str = "guidelines") -> str:  # type: ignore
        return (
            "⚠️ The retrieval pipeline could not be loaded. "
            "Run `python Build_Index.py` to build the FAISS index, then restart the app."
        )

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Nephrology Clinical Evidence Assistant",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CSS — Apple/Gemini-inspired: clean whitespace, calm clinical palette,
# smooth transitions, large tap targets for ward use on tablets
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {
            --teal:          #0D6E6B;
            --teal-dark:     #094E4C;
            --teal-hover:    #EBF5F5;
            --teal-light:    #E0F2F1;
            --bg:            #F5F7FA;
            --surface:       #FFFFFF;
            --border:        #E0E7E9;
            --text-primary:  #1A3533;
            --text-muted:    #52706E;
            --amber:         #A85A00;
            --amber-bg:      #FFF4E6;
            --amber-border:  #F4C87A;
            --error-bg:      #FEF2F2;
            --error-text:    #991B1B;
            --success-bg:    #ECFDF5;
            --success-text:  #065F46;
            --shadow-sm:     0 1px 4px rgba(0,0,0,0.06);
            --shadow-md:     0 4px 16px rgba(13,110,107,0.12);
            --radius:        12px;
            --radius-sm:     8px;
        }

        /* ── Global reset ── */
        html, body, .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg) !important;
            color: var(--text-primary);
        }

        /* Hide default Streamlit menu and footer, but keep header/sidebar controls functional */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        /* Ensure sidebar toggle button is always visible and styled */
        [data-testid="stSidebarCollapseButton"], 
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] button {
            visibility: visible !important;
            color: var(--teal) !important;
        }

        /* Fix Streamlit's own text colour in all contexts */
        p, span, label, li, div, h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ── App header ── */
        .app-header {
            background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%);
            padding: 1.4rem 1.8rem;
            border-radius: var(--radius);
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-md);
        }
        .app-header h1 {
            margin: 0 0 0.25rem 0;
            font-size: 1.35rem;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .app-header .subtitle {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(255,255,255,0.85);
            font-weight: 400;
        }

        /* ── Status pill ── */
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.62rem;
            font-weight: 600;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            vertical-align: middle;
        }
        .pill-ready { background: rgba(255,255,255,0.22); color: #FFFFFF; }
        .pill-error { background: #FCE4E4; color: #B02A2A; }

        /* ── Source badge ── */
        .source-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.65rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .source-badge-guidelines {
            background: var(--teal-light);
            color: var(--teal-dark);
        }
        .source-badge-publications {
            background: #EDE9FE;
            color: #5B21B6;
        }
        .source-badge-both {
            background: #FEF3C7;
            color: #92400E;
        }

        /* ── Disclaimer banner ── */
        .disclaimer {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            background: var(--amber-bg);
            border: 1px solid var(--amber-border);
            border-radius: var(--radius-sm);
            padding: 0.65rem 1rem;
            margin-bottom: 1.1rem;
            font-size: 0.8rem;
            color: var(--amber);
            line-height: 1.5;
        }
        .disclaimer strong { color: var(--amber); }

        /* ── Quick-topic chips (main area) ── */
        .chip-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.5rem;
        }
        .chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 1.1rem;
        }
        .chip {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.42rem 0.9rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }
        .chip:hover {
            border-color: var(--teal);
            color: var(--teal-dark);
            background: var(--teal-hover);
        }

        /* ── Empty state ── */
        .empty-state {
            text-align: center;
            padding: 3rem 1.5rem;
        }
        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 0.75rem;
        }
        .empty-state h2 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.35rem;
        }
        .empty-state p {
            font-size: 0.88rem;
            color: var(--text-muted);
            max-width: 480px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ── Chat bubbles ── */
        div[data-testid="stChatMessage"] {
            background: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            margin-bottom: 0.5rem;
        }
        /* Assistant bubble — subtle left accent */
        div[data-testid="stChatMessage"][data-testid*="assistant"],
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--teal);
        }

        /* ── Chat input ── */
        .stChatInput textarea {
            border-radius: var(--radius-sm) !important;
            border-color: var(--border) !important;
            font-size: 0.93rem !important;
            padding: 0.75rem 1rem !important;
            background: var(--surface) !important;
            color: var(--text-primary) !important;
        }
        .stChatInput textarea:focus {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 3px rgba(13,110,107,0.12) !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--teal-dark);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin: 1.2rem 0 0.4rem 0;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] li {
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.6;
        }
        section[data-testid="stSidebar"] a {
            color: var(--teal) !important;
            text-decoration: none;
        }
        section[data-testid="stSidebar"] a:hover { text-decoration: underline; }

        /* Sidebar sample question buttons */
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 0.8rem !important;
            font-weight: 400 !important;
            text-align: left !important;
            width: 100% !important;
            padding: 0.5rem 0.75rem !important;
            margin-bottom: 0.3rem !important;
            transition: all 0.15s ease !important;
            line-height: 1.4 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            border-color: var(--teal) !important;
            color: var(--teal-dark) !important;
            background: var(--teal-hover) !important;
        }

        /* Clear chat button */
        .clear-btn button {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            color: var(--text-muted) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 0.78rem !important;
            padding: 0.3rem 0.7rem !important;
            transition: all 0.15s ease !important;
        }
        .clear-btn button:hover {
            border-color: #E11D48 !important;
            color: #E11D48 !important;
            background: #FFF1F3 !important;
        }

        /* KDIGO chapter navigator in sidebar */
        .chapter-item {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            padding: 0.4rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .chapter-num {
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--teal);
            min-width: 1.2rem;
        }

        /* Spinner override */
        .stSpinner > div {
            border-top-color: var(--teal) !important;
        }

        /* Error / warning boxes */
        .stAlert {
            border-radius: var(--radius-sm) !important;
        }

        /* ── Source selector radio styling ── */
        div[data-testid="stRadio"] > label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: var(--teal-dark) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.07em !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sample questions — context-aware based on source selection
# --------------------------------------------------------------------------
SAMPLE_QUESTIONS_GUIDELINES = [
    "What are the KDIGO 2026 criteria for diagnosing AKI in adults?",
    "What are the 3 stages of AKI by serum creatinine and urine output?",
    "What is the multicomponent kidney protection strategy for surgical patients?",
    "How do I select a baseline creatinine to diagnose AKI?",
    "When should I initiate renal replacement therapy in AKI?",
    "Which drugs should I avoid or monitor in patients at risk of AKI?",
    "What is the approach to contrast-associated AKI prevention?",
    "How should a patient be followed up after hospital discharge with AKI?",
    "What are the KDIGO criteria for Acute Kidney Disease (AKD)?",
    "How is AKI staged in neonates?",
]

SAMPLE_QUESTIONS_PUBLICATIONS = [
    "What did the STARRT-AKI trial find about early vs late RRT?",
    "What is the evidence for restrictive vs liberal fluid strategy in AKI?",
    "What are the latest findings on vancomycin-induced nephrotoxicity?",
    "What do recent meta-analyses say about biomarker-guided AKI management?",
    "What is the evidence for SGLT2 inhibitors in AKI prevention?",
]

SAMPLE_QUESTIONS_BOTH = [
    "What does KDIGO recommend for RRT timing, and what do the trials show?",
    "What is the guideline and research evidence on contrast-associated AKI?",
    "How do guidelines and recent studies compare on fluid management in AKI?",
    "What do guidelines and publications say about nephrotoxin stewardship?",
    "What is the evidence base behind the multicomponent kidney protection strategy?",
]

# --------------------------------------------------------------------------
# KDIGO chapters for sidebar navigator
# --------------------------------------------------------------------------
KDIGO_CHAPTERS = [
    ("1", "AKI & AKD Definition, Identification, Classification"),
    ("2", "AKI Prediction, Risk Stratification & Diagnosis"),
    ("3", "Prevention & General Management of AKI"),
    ("4", "Drug- & Nephrotoxin-Associated AKI"),
    ("5", "Kidney Replacement Therapy (RRT)"),
    ("6", "Recovery, Follow-up & Long-term Outcomes"),
]

# --------------------------------------------------------------------------
# Source type display configuration
# --------------------------------------------------------------------------
SOURCE_CONFIG = {
    "guidelines": {
        "icon": "📋",
        "label": "Guidelines",
        "description": "KDIGO clinical practice guidelines",
        "spinner": "Searching KDIGO 2026 guideline...",
        "badge_class": "source-badge-guidelines",
    },
    "publications": {
        "icon": "📄",
        "label": "Publications",
        "description": "Peer-reviewed research papers & clinical trials",
        "spinner": "Searching research publications...",
        "badge_class": "source-badge-publications",
    },
    "both": {
        "icon": "📚",
        "label": "Both",
        "description": "Guidelines + publications combined",
        "spinner": "Searching all clinical evidence...",
        "badge_class": "source-badge-both",
    },
}

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    # Branding
    st.markdown(
        "<p style='font-size:1rem;font-weight:700;color:#0D6E6B;margin-bottom:0.1rem;'>🫘 Nephrology Evidence Assistant</p>"
        "<p style='font-size:0.75rem;color:#52706E;margin-top:0;'>Guidelines & Publications</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Evidence Source Selector ──
    st.markdown("<div class='sidebar-section-label'>📚 Evidence Source</div>", unsafe_allow_html=True)

    # Build radio options based on what indexes are available
    source_options = []
    source_values = []

    if "guidelines" in AVAILABLE_SOURCES:
        source_options.append("📋 Guidelines")
        source_values.append("guidelines")
    if "publications" in AVAILABLE_SOURCES:
        source_options.append("📄 Publications")
        source_values.append("publications")
    if len(AVAILABLE_SOURCES) >= 2:
        source_options.append("📚 Both")
        source_values.append("both")

    # Fallback when no indexes are built yet
    if not source_options:
        source_options = ["📋 Guidelines"]
        source_values = ["guidelines"]

    selected_label = st.radio(
        "Select evidence source",
        source_options,
        index=0,
        label_visibility="collapsed",
        key="source_selector",
    )
    selected_source = source_values[source_options.index(selected_label)]

    # Show description of selected source
    cfg = SOURCE_CONFIG[selected_source]
    st.caption(cfg["description"])

    # Show availability status
    if "publications" not in AVAILABLE_SOURCES:
        st.markdown(
            "<p style='font-size:0.75rem;color:#A85A00;margin-top:0.3rem;'>"
            "ℹ️ No publications indexed yet. Add PDFs to <code>documents/publications/</code> "
            "and run <code>python Build_Index.py</code></p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # About — dynamic based on source selection
    st.markdown("<div class='sidebar-section-label'>About</div>", unsafe_allow_html=True)
    if selected_source == "guidelines":
        st.markdown(
            "<p>Answers are retrieved <strong>exclusively</strong> from the "
            "<em>KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) "
            "and Acute Kidney Disease (AKD)</em> — Public Review Draft, March 2026. "
            "Every answer cites the source page.</p>",
            unsafe_allow_html=True,
        )
    elif selected_source == "publications":
        st.markdown(
            "<p>Answers are retrieved from <strong>indexed research publications</strong> "
            "(clinical trials, meta-analyses, reviews). "
            "Every answer cites the source paper and page.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p>Answers synthesize evidence from <strong>both</strong> the KDIGO 2026 guideline "
            "<strong>and</strong> indexed research publications. "
            "Each claim is tagged as guideline or research evidence.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # KDIGO Chapter Navigator — show only when guidelines or both is selected
    if selected_source in ("guidelines", "both"):
        st.markdown("<div class='sidebar-section-label'>Guideline Chapters</div>", unsafe_allow_html=True)
        for num, title in KDIGO_CHAPTERS:
            st.markdown(
                f"<div class='chapter-item'>"
                f"<span class='chapter-num'>Ch.{num}</span>"
                f"<span>{title}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # Sample questions — dynamic based on source selection
    st.markdown("<div class='sidebar-section-label'>Sample Questions</div>", unsafe_allow_html=True)
    st.caption("Tap to ask the assistant")

    if selected_source == "publications":
        sample_qs = SAMPLE_QUESTIONS_PUBLICATIONS
    elif selected_source == "both":
        sample_qs = SAMPLE_QUESTIONS_BOTH
    else:
        sample_qs = SAMPLE_QUESTIONS_GUIDELINES

    for q in sample_qs:
        if st.button(q, key=f"sq_{selected_source}_{hash(q)}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")

    # Limitations
    st.markdown("<div class='sidebar-section-label'>Limitations</div>", unsafe_allow_html=True)
    st.markdown(
        "<ul style='padding-left:1rem;margin:0;'>"
        "<li>Answers sourced only from the loaded PDFs (guidelines and/or publications)</li>"
        "<li>Not a substitute for nephrology consultation</li>"
        "<li>Figure/flowchart descriptions are annotated summaries — verify against original</li>"
        "<li>Always cite and verify the source page before clinical use</li>"
        "</ul>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # How it works — collapsed by default (not distracting in a clinical setting)
    with st.expander("How does the retrieval work?", expanded=False):
        st.markdown(
            "<ol style='padding-left:1rem;margin:0;font-size:0.8rem;color:#52706E;'>"
            "<li>PDFs (guidelines and/or publications) are split into structured chunks (tables preserved as Markdown).</li>"
            "<li>Your question is rephrased 3 ways to catch clinical synonyms.</li>"
            "<li>The best chunks are retrieved by semantic similarity from the selected source, then reranked.</li>"
            "<li>The LLM answers <em>only</em> from those chunks and cites the page.</li>"
            "</ol>",
            unsafe_allow_html=True,
        )

    # Clear conversation
    st.markdown("---")
    st.markdown("<div class='sidebar-section-label'>Session</div>", unsafe_allow_html=True)
    st.markdown("<div class='clear-btn'>", unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation", key="clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built by Sunil Kumar Nallani")
    st.caption("[LinkedIn](https://www.linkedin.com/in/nallani-sunil-kumar-67227a243/) · [GitHub](https://github.com/sunilkumar-nallani/Bacteremia_Clinical_Guideline_Assistant)")

# --------------------------------------------------------------------------
# Main area — Header (dynamic based on source selection)
# --------------------------------------------------------------------------
status_pill = (
    '<span class="pill pill-ready">● Live</span>'
    if PIPELINE_READY
    else '<span class="pill pill-error">● Pipeline Error</span>'
)

# Dynamic header text based on source
if selected_source == "guidelines":
    header_title = "🫘 KDIGO 2026 &nbsp; AKI / AKD Guideline Assistant"
    header_subtitle = (
        "Evidence-based answers from the KDIGO 2026 Clinical Practice Guideline for "
        "Acute Kidney Injury &amp; Acute Kidney Disease — cited to source, chapter, and page."
    )
elif selected_source == "publications":
    header_title = "📄 Nephrology Research &nbsp; Publications Assistant"
    header_subtitle = (
        "Evidence from indexed peer-reviewed publications, clinical trials, and meta-analyses "
        "— cited to source paper and page."
    )
else:
    header_title = "📚 Nephrology Evidence &nbsp; Assistant"
    header_subtitle = (
        "Comprehensive answers from both KDIGO 2026 guidelines and indexed research publications "
        "— each claim tagged with its evidence source."
    )

st.markdown(
    f"""
    <div class="app-header">
        <h1>{header_title} &nbsp;{status_pill}</h1>
        <p class="subtitle">{header_subtitle}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Disclaimer — always visible, non-intrusive
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer">
        <span style="font-size:1rem;">⚠️</span>
        <span>
            <strong>Clinical decision support tool — not a substitute for clinical judgment or nephrology consultation.</strong>
            This tool retrieves information from indexed clinical documents. Recommendations may change in
            final published versions. Always verify cited content against the source document before clinical use.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Pipeline error banner
# --------------------------------------------------------------------------
if not PIPELINE_READY:
    st.error(
        f"**Pipeline failed to load:** `{PIPELINE_ERROR}`  \n"
        "Run `python Build_Index.py` to build the FAISS index, then restart the app."
    )

# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "👨‍⚕️" if message["role"] == "user" else "🫘"
    with st.chat_message(message["role"], avatar=avatar):
        # Show source badge for assistant messages
        if message["role"] == "assistant" and "source_type" in message:
            src = message["source_type"]
            src_cfg = SOURCE_CONFIG.get(src, SOURCE_CONFIG["guidelines"])
            st.markdown(
                f'<span class="source-badge {src_cfg["badge_class"]}">'
                f'{src_cfg["icon"]} {src_cfg["label"]}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(message["content"])

# --------------------------------------------------------------------------
# Empty state — shown when no conversation yet
# --------------------------------------------------------------------------
if not st.session_state.messages:
    if selected_source == "guidelines":
        empty_text = (
            "This assistant searches the KDIGO 2026 guideline and returns "
            "evidence-grounded answers with exact page citations. "
            "Use the sample questions in the sidebar, or type your own below."
        )
    elif selected_source == "publications":
        empty_text = (
            "This assistant searches indexed research publications and returns "
            "evidence-grounded answers with citations to source papers and pages. "
            "Use the sample questions in the sidebar, or type your own below."
        )
    else:
        empty_text = (
            "This assistant searches both KDIGO 2026 guidelines and indexed publications, "
            "returning comprehensive answers with evidence sources clearly tagged. "
            "Use the sample questions in the sidebar, or type your own below."
        )

    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">🫘</div>
            <h2>Ask a clinical question about AKI or AKD</h2>
            <p>{empty_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Input — typed or triggered by sidebar sample-question chip
# --------------------------------------------------------------------------
typed_question = st.chat_input(
    "Ask about AKI staging, RRT initiation, fluid management, nephrotoxins, follow-up..."
)

question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👨‍⚕️"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🫘"):
        # Show source badge
        src_cfg = SOURCE_CONFIG[selected_source]
        st.markdown(
            f'<span class="source-badge {src_cfg["badge_class"]}">'
            f'{src_cfg["icon"]} {src_cfg["label"]}</span>',
            unsafe_allow_html=True,
        )

        with st.spinner(src_cfg["spinner"]):
            try:
                answer = answer_question(question, source_type=selected_source)
            except openai.InternalServerError:
                st.error("⚠️ The institutional API server (GWDG) is currently overloaded or restarting. Please wait a moment and try your question again.")
                st.stop()
            except openai.AuthenticationError:
                st.error("⚠️ API connection failed. Please verify the environment credentials.")
                st.stop()
            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred in the retrieval pipeline: {str(e)}")
                st.stop()
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source_type": selected_source,
    })
