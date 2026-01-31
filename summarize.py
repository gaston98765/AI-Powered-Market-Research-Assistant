# summarize.py

from typing import List, Tuple
import json
import re
from ollama import chat as ollama_chat


def call_business_agent(
    model: str,
    user_message: str,
    chat_history: List[dict],
    context_text: str,
) -> str:
    system_prompt = """
You are a senior business & market analysis consultant.
Respond with a clean, scannable layout using concise bullet points.

OUTPUT FORMAT (strict):

### Executive Summary
- 3 to 6 short bullets with crisp takeaways.

### SWOT Analysis
**Strengths**
- bullet 1
- bullet 2
**Weaknesses**
- bullet 1
- bullet 2
**Opportunities**
- bullet 1
- bullet 2
**Threats**
- bullet 1
- bullet 2

### Strategic Recommendations
- 3 to 7 actionable, imperative bullets (start with verbs).

Rules: Avoid long paragraphs. Prefer bullets. If info is limited, state assumptions.
If extra text is provided, use it without mentioning the word "context".
Do NOT output JSON.
""".strip()

    if context_text and context_text.strip():
        user_content = (
            f"{user_message}\n\n"
            "Here is additional information that may be relevant. "
            "Use it only if it helps answer the question:\n"
            f"{context_text}"
        )
    else:
        user_content = user_message

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_content})

    resp = ollama_chat(model=model, messages=messages)
    return resp["message"]["content"]


def extract_top_companies(model: str, texts: List[str], topic: str) -> List[dict]:
    combined_text = "\n\n".join(texts)[:8000]

    prompt = f"""
You are a business intelligence analyst.

You will receive:
- A topic: "{topic}"
- A body of text that may mention many companies.

Your task:
1. Identify the most relevant or frequently mentioned companies.
2. Roughly count mentions (or prominence).
3. Rank from most to least important.
4. Return ONLY a JSON array (no extra text) like:

[
  {{"company": "CompanyName", "mentions": 25}},
  {{"company": "AnotherCompany", "mentions": 18}}
]

CONTENT START
{combined_text}
CONTENT END
""".strip()

    resp = ollama_chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract and rank top companies from text. "
                    "Respond with JSON ONLY, no commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp["message"]["content"]
    # Parse JSON array within the model output, be lenient with extra text
    data: List[dict] = []
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            # Fall back to empty list on failure
            data = []
    except Exception:
        data = []

    # Ensure normalized structure and then recompute mentions from the actual text
    def strip_suffixes(name: str) -> str:
        return re.sub(r",?\s+(Inc\.|Incorporated|Corp\.|Corporation|Ltd\.?|LLC|PLC|S\.?A\.?)$", "", name, flags=re.IGNORECASE)

    def count_mentions(text: str, name: str) -> int:
        if not name or not text:
            return 0
        variants = [name]
        base = strip_suffixes(name)
        if base and base.lower() != name.lower():
            variants.append(base)
        counts = []
        for v in {v.strip(): None for v in variants}.keys():
            if not v:
                continue
            pattern = r"(?i)(?<!\w)" + re.escape(v) + r"(?!\w)"
            counts.append(len(re.findall(pattern, text)))
        return max(counts) if counts else 0

    cleaned = []
    for item in data:
        if isinstance(item, dict) and "company" in item:
            company_name = str(item.get("company", "Unknown")).strip()
            computed_mentions = count_mentions(combined_text, company_name)
            cleaned.append({
                "company": company_name or "Unknown",
                "mentions": int(computed_mentions),
            })

    # Sort by mentions desc and keep top 15
    cleaned.sort(key=lambda x: x.get("mentions", 0), reverse=True)
    return cleaned[:15]
