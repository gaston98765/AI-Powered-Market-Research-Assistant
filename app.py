import os
import json
import re
from typing import Optional
import PyPDF2
import pandas as pd
import streamlit as st
from fpdf import FPDF
from fpdf.errors import FPDFException
from ollama import chat as ollama_chat

from scraper import find_urls_in_text, scrape_url
from summarize import call_business_agent, extract_top_companies

HISTORY_FILE = "memory.json"
CHAT_HISTORY_FILE = "chat_history.json"


# ---------- File helpers ----------

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[json] load error for {path}:", e)
    return default


def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[json] save error for {path}:", e)


def extract_text_and_meta(uploaded_file, max_chars=6000):
    """Return (text, meta) for PDF or text uploads. Meta includes title/author/pages if available."""
    if not uploaded_file:
        return "", {}
    text = ""
    meta = {}
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            info = getattr(reader, "metadata", None) or getattr(reader, "documentInfo", None) or {}
            # Normalize common PDF metadata keys
            meta = {
                "title": info.get("/Title") or info.get("Title") if isinstance(info, dict) else None,
                "author": info.get("/Author") or info.get("Author") if isinstance(info, dict) else None,
                "pages": len(reader.pages),
            }
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print("[upload] read error:", e)
    return (text[: max_chars], {k: v for k, v in meta.items() if v})

# ---------- History helpers ----------

def load_history():
    return load_json(HISTORY_FILE, [])


def add_to_history(topic, summary, leaderboard):
    history = load_history()
    history.append({"topic": topic, "summary": summary, "leaderboard": leaderboard})
    save_json(HISTORY_FILE, history)


def load_chat_history():
    return load_json(CHAT_HISTORY_FILE, [])


def save_chat_history(messages):
    save_json(CHAT_HISTORY_FILE, messages)


# ---------- Deep analysis detector ----------

def is_deep_analysis_request(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()

    greetings = [
        "hello", "hi", "hey", "salut", "yo",
        "good morning", "good evening", "good afternoon",
    ]
    if t in greetings or any(t.startswith(g + " ") for g in greetings):
        return False

    triggers = [
        "analyze", "analyse", "analysis",
        "swot", "business plan",
        "detailed", "in detail", "deep dive", "full report", "full analysis",
        "market analysis", "competitor analysis", "competitive analysis",
        "strategy for", "strategic plan", "go-to-market",
        "market entry strategy", "expansion strategy",
    ]
    return any(phrase in t for phrase in triggers)


def wants_doc_context(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    keywords = ["uploaded", "upload", "pdf", "document", "file", "this file", "what's in", "summarize file", "read my", "read the"]
    return any(k in t for k in keywords)


# ---------- PDF report ----------

def generate_pdf_report(topic, summary, leaderboard):
    def sanitize_for_pdf(text: str) -> str:
        """Replace unsupported unicode (e.g., •) with ASCII so core fonts work.
        Falls back to Latin-1 safe text for FPDF core fonts (Helvetica).
        """
        if not text:
            return ""
        replacements = {
            "•": "- ",
            "◦": "- ",
            "‣": "- ",
            "∙": "- ",
            "●": "- ",
            "○": "- ",
            "–": "-",
            "—": "-",
            "‑": "-",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            " ": " ",  # non-breaking space
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        try:
            # Ensure it is encodable by FPDF core fonts
            text.encode("latin-1")
            return text
        except Exception:
            import unicodedata
            return (
                unicodedata.normalize("NFKD", text)
                .encode("latin-1", "ignore")
                .decode("latin-1")
            )
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    page_width = pdf.w - 2 * pdf.l_margin

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Business AI Market Report", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(page_width, 8, f"Topic / Prompt: {sanitize_for_pdf(topic)}")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Executive Summary, SWOT & Recommendations", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "", 11)
    # Normalize bullets to line-per-item for PDF and sanitize unicode
    formatted_summary = format_bullets_markdown(summary)
    pdf.multi_cell(page_width, 6, sanitize_for_pdf(formatted_summary) or "(No summary available)")

    if leaderboard:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Top Companies", ln=True)
        pdf.ln(1)
        pdf.set_font("Arial", "", 11)

        for row in leaderboard:
            company = str(row.get("company", "Unknown"))
            mentions = str(row.get("mentions", "N/A"))
            line = sanitize_for_pdf(f"- {company} (mentions: {mentions})")
            if len(line) > 200:
                line = line[:197] + "..."
            try:
                pdf.multi_cell(page_width, 6, line)
            except FPDFException:
                pdf.cell(0, 6, f"- {company}", ln=True)

    pdf_output = pdf.output(dest="S")
    return bytes(pdf_output) if isinstance(pdf_output, (bytes, bytearray)) else pdf_output.encode("latin-1")


# ---------- Streamlit base config ----------

st.set_page_config(page_title="Business AI Chatbot", layout="wide")

st.sidebar.title("Business AI Chatbot")
view = st.sidebar.radio("View", ["Chat", "History"])

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT (optional)",
    type=["pdf", "txt"],
    help="The bot will use this as context when you ask questions.",
)

if "doc_text" not in st.session_state:
    st.session_state["doc_text"] = ""

if uploaded_file is not None:
    text, meta = extract_text_and_meta(uploaded_file)
    if text:
        st.session_state["doc_text"] = text
        st.session_state.setdefault("doc_info", {}).update(meta or {})
        st.sidebar.success("Document loaded into memory ✅")
    else:
        st.sidebar.warning("Could not read this file.")

if st.session_state.get("doc_info"):
    info = st.session_state["doc_info"]
    parts = [f"{k.title()}: {v}" for k, v in info.items()]
    st.sidebar.info("Uploaded document — " + "; ".join(parts))

st.sidebar.checkbox(
    "Always use uploaded document",
    value=False,
    key="use_doc_auto",
    help="Include the uploaded file in answers even if you don't mention it.",
)

st.sidebar.caption(
    "Tip: paste a URL in your question, e.g.\n"
    "`Analyze https://example.com and give me a SWOT of their product.`"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None


# ---------- Performance helpers ----------
def last_messages(messages, n=6):
    """Return only the last n chat messages (for faster prompts)."""
    try:
        return messages[-n:]
    except Exception:
        return messages


# ---------- Formatting helpers ----------
def format_bullets_markdown(text: str) -> str:
    """Ensure bullet items appear one per line and normalize headings.
    - Converts inline "•" separators into real newline bullets.
    - Normalizes heading + inline bullet patterns into a heading line followed by bullets.
    """
    if not text:
        return text
    t = text
    # Convert inline bullets into newline list items
    t = re.sub(r"\s*•\s*", "\n- ", t)
    # If a SWOT heading is followed by a dash item on same line, break the line
    t = re.sub(r"(?im)(^\*?\*?(Strengths|Weaknesses|Opportunities|Threats)\*?\*?\s*:?)(\s*-\s*)", r"\1\n- ", t)
    # Bold plain headings appearing alone on a line
    t = re.sub(r"(?im)^(Strengths|Weaknesses|Opportunities|Threats)\s*$", r"**\1**", t)
    # Collapse any duplicate bullet markers after newline
    t = re.sub(r"\n[\-•]\s*[\-•]\s*", "\n- ", t)
    return t


def strip_top_companies_section(text: str) -> str:
    """Remove any 'Top Companies' section (heading + bullet list) from text to avoid duplicates.
    Matches markdown headings like '### Top Companies' or a plain 'Top Companies' line,
    then consumes following bullet lines. Keeps the rest intact.
    """
    if not text:
        return text
    pattern = re.compile(
        r"(?ims)^[ \t]*(?:#{1,6}[ \t]*)?Top Companies[ \t]*\n(?:^[ \t]*(?:-|\*|\d+\.)[ \t].*\n?)*",
    )
    return re.sub(pattern, "", text)


# ==================== Message Prep Helper ====================

def prepare_messages(
    *,
    mode: str,
    user_message: str,
    include_history: bool = True,
    use_doc_context: bool = False,
    doc_text: str = "",
    doc_meta: Optional[dict] = None,
):
    """Build messages for ollama_chat.
    mode: "chat" | "analysis" decides the system prompt.
    - Caps history with last_messages.
    - Adds a short document excerpt when requested in chat mode.
    """
    if mode == "chat":
        system = (
            "You are a helpful business assistant. "
            "Answer naturally and clearly. "
            "Use the uploaded document only if relevant. "
            "Do not produce SWOT or long reports unless asked."
        )
    else:
        # The analysis prompt lives in summarize.call_business_agent
        system = "You are a helpful assistant."

    messages = [{"role": "system", "content": system}]

    if include_history:
        messages.extend(last_messages(st.session_state["messages"][:-1], 6))

    # Optional short doc context for chat mode
    if use_doc_context and doc_text:
        meta_str = "; ".join(f"{k}: {v}" for k, v in (doc_meta or {}).items())
        excerpt = doc_text[:1500]
        messages.append({
            "role": "user",
            "content": f"The user uploaded a document ({meta_str}). Use it to answer: {excerpt}",
        })

    messages.append({"role": "user", "content": user_message})
    return messages


# ==================== CHAT VIEW ====================

if view == "Chat":
    st.title(" Business Analysis Chat")
    st.write(
        "- Simple questions → normal chat response.\n"
        "- Requests like *'analyze in detail'*, *'do a SWOT'*, *'business plan'* → "
        "full analysis with SWOT, top companies chart, and PDF."
    )

    model = st.selectbox(
        "Local model (Ollama tag):",
        ["llama3.2:3b", "phi3:mini", "qwen2.5:3b"],
        index=0,
    )

    user_message = st.chat_input("Type your question or message here...")

    if user_message:
        st.session_state["messages"].append({"role": "user", "content": user_message})

        if not is_deep_analysis_request(user_message):
            # --- Normal chat ---
            with st.spinner("Thinking..."):
                use_doc_context_flag = (
                    (wants_doc_context(user_message) or st.session_state.get("use_doc_auto", False))
                    and bool(st.session_state.get("doc_text"))
                )
                messages = prepare_messages(
                    mode="chat",
                    user_message=user_message,
                    include_history=True,
                    use_doc_context=use_doc_context_flag,
                    doc_text=st.session_state.get("doc_text", ""),
                    doc_meta=st.session_state.get("doc_info") or {},
                )
                resp = ollama_chat(model=model, messages=messages)
                assistant_text = resp["message"]["content"]

            st.session_state["messages"].append(
                {"role": "assistant", "content": assistant_text}
            )
            save_chat_history(st.session_state["messages"])

        else:
            # --- Deep business analysis ---
            summary = ""
            leaderboard = []

            with st.spinner("Thinking as a business analyst..."):
                # Build context (uploaded doc + URLs)
                context_parts = []
                doc_text = st.session_state.get("doc_text", "")
                doc_relevant = wants_doc_context(user_message) or st.session_state.get("use_doc_auto", False)

                # Only include uploaded document if the user asked to use it
                if doc_relevant and doc_text:
                    context_parts.append("Text from uploaded document:\n" + doc_text[:6000])

                urls = find_urls_in_text(user_message)
                scraped_texts = []
                for url in urls:
                    page_text = scrape_url(url)
                    if page_text:
                        scraped_texts.append(page_text)
                        context_parts.append(f"Text from web page {url}:\n{page_text[:3000]}")

                context_text = "\n\n".join(context_parts)

                # 1) Full analysis
                summary = call_business_agent(
                    model=model,
                    user_message=user_message,
                    chat_history=last_messages(st.session_state["messages"][:-1], 6),
                    context_text=context_text,
                )
                # Remove any Top Companies that model might have inserted
                summary = strip_top_companies_section(summary)

                # 2) Top companies
                # Build corpus for company extraction and mention counting
                # Always include the generated summary and the user's prompt,
                # then add any uploaded doc excerpt (when relevant) and scraped pages.
                texts_for_companies = []
                texts_for_companies.append(summary)
                texts_for_companies.append(user_message)
                if doc_relevant and doc_text:
                    texts_for_companies.append(doc_text[:6000])
                texts_for_companies.extend(t[:6000] for t in scraped_texts)

                leaderboard = extract_top_companies(model, texts_for_companies, user_message)

                # The analysis text already contains headings; normalize bullets for readability
                response_markdown = format_bullets_markdown(summary) + "\n\n"
                if leaderboard:
                    response_markdown += "### Top Companies\n\n"
                    for row in leaderboard:
                        company = row.get("company", "Unknown")
                        mentions = row.get("mentions", "N/A")
                        response_markdown += f"- **{company}** (mentions: {mentions})\n"

            # Store leaderboard on the message so its chart renders WITH the answer
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_markdown, "leaderboard": leaderboard}
            )

            st.session_state["last_analysis"] = {
                "topic": user_message,
                "summary": summary, 
                "leaderboard": leaderboard,
            }

            add_to_history(user_message, summary, leaderboard)
            save_chat_history(st.session_state["messages"])

    # Render chat. If a message has a leaderboard attached, draw its chart here
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("leaderboard"):
                df_chart = pd.DataFrame(msg["leaderboard"])
                if {"company", "mentions"}.issubset(df_chart.columns):
                    df_chart["mentions"] = pd.to_numeric(df_chart["mentions"], errors="coerce")
                    df_chart = df_chart.dropna(subset=["mentions"])
                    if not df_chart.empty:
                        st.markdown("#### 📊 Top Companies (Mentions)")
                        st.bar_chart(df_chart.set_index("company")["mentions"])

    # Keep PDF controls tied to the most recent analysis
    last = st.session_state["last_analysis"]

    # PDF download
    if last and (last.get("summary") or last.get("leaderboard")):
        pdf_bytes = generate_pdf_report(
            last["topic"], last["summary"], last["leaderboard"]
        )
        st.download_button(
            label="📄 Download last analysis as PDF",
            data=pdf_bytes,
            file_name="business_ai_report.pdf",
            mime="application/pdf",
        )


# ==================== HISTORY VIEW ====================

elif view == "History":
    st.title(" Full History (Discussion + Analyses)")

    if st.button(" Clear Entire History"):
        for path in (CHAT_HISTORY_FILE, HISTORY_FILE):
            if os.path.exists(path):
                os.remove(path)
        st.session_state["messages"] = []
        st.session_state["last_analysis"] = None
        st.success("All history cleared.")
        st.rerun()

    chat_history = load_chat_history()
    analysis_history = load_history()

    if not chat_history and not analysis_history:
        st.info("No history yet. Ask something in the Chat tab first.")
    else:
        with st.container():
            if analysis_history:
                st.markdown("#####  Analyses (Summary + Companies)")
                for i, item in enumerate(reversed(analysis_history)):
                    topic = item.get("topic", f"Analysis {i+1}")
                    summary = item.get("summary", "")
                    leaderboard = item.get("leaderboard", [])

                    with st.expander(topic):
                        st.markdown("**Summary & SWOT / Recommendations**")
                        st.markdown(summary)

                        if leaderboard:
                            st.markdown("**Top Companies (Table)**")
                            df = pd.DataFrame(leaderboard)
                            st.dataframe(df)

                            if {"company", "mentions"}.issubset(df.columns):
                                df_chart = df.copy()
                                df_chart["mentions"] = pd.to_numeric(
                                    df_chart["mentions"], errors="coerce"
                                )
                                df_chart = df_chart.dropna(subset=["mentions"])
                                if not df_chart.empty:
                                    st.markdown("** Mentions Bar Chart**")
                                    st.bar_chart(df_chart.set_index("company")["mentions"])
                        else:
                            st.write("_No companies data stored for this analysis._")
            else:
                st.info("No analyses stored yet.")
