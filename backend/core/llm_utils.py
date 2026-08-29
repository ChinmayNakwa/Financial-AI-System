# backend/core/llm_utils.py

from datetime import datetime


def llm_text(response) -> str:
    """Normalize a LangChain chat response's ``.content`` to a plain string.

    langchain-google-genai (Gemini 3.x) returns ``content`` as a list of
    content blocks, e.g. ``[{"type": "text", "text": "..."}]``, instead of a
    bare string. Callers that run regex/JSON parsing over the text need a
    single string regardless of the provider's shape.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return "".join(parts)

    return str(content)


def current_date_str() -> str:
    """Human-readable current date, e.g. 'August 29, 2026'.

    Used to keep the recency reasoning in the RAG prompts anchored to the
    real date instead of a hardcoded month.
    """
    return datetime.now().strftime("%B %d, %Y")
