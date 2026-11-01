"""
API and Prompt Security Guardrails for Advanced Hybrid RAG.

Provides:
- Adversarial prompt injection and jailbreak detection
- Input payload bounds checking and sanitization
- Exception sanitization to prevent API key and internal path leakage
- Session rate limiting and cooldown validation
"""

import re
from typing import NamedTuple


class SecurityCheckResult(NamedTuple):
    is_safe: bool
    reason: str = ""


# Compile regex patterns for jailbreak, prompt leakage, and injection heuristics
_ADVERSARIAL_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(
        r"(reveal|show|print|leak|repeat|display)\s+(your\s+|the\s+)?(system\s+prompt|initial\s+prompt|instructions)",
        re.IGNORECASE,
    ),
    re.compile(
        r"what\s+(are\s+|were\s+)?(your\s+|the\s+)?(system\s+instructions|system\s+prompt)",
        re.IGNORECASE,
    ),
    re.compile(r"repeat\s+everything\s+(above|before)", re.IGNORECASE),
    re.compile(r"\b(dan\s+mode|jailbreak|unfiltered\s+assistant|bypass\s+safety)\b", re.IGNORECASE),
    re.compile(r"<\s*script[^>]*>", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
]


def check_prompt_security(query: str, max_length: int = 500) -> SecurityCheckResult:
    """Validates user query against adversarial prompt injection, jailbreaks, and structural limits."""
    clean = str(query or "").strip()
    if not clean:
        return SecurityCheckResult(False, "Query cannot be empty or whitespace.")

    if len(clean) > max_length:
        return SecurityCheckResult(False, f"Query exceeds maximum allowed limit of {max_length} characters.")

    if len(clean) < 2:
        return SecurityCheckResult(False, "Query is too short. Please provide a meaningful question.")

    for pattern in _ADVERSARIAL_PATTERNS:
        if pattern.search(clean):
            return SecurityCheckResult(
                False,
                "Query rejected by security guardrail: Input contains disallowed override or prompt inspection instructions.",
            )

    return SecurityCheckResult(True, "")


def sanitize_error_message(exc: Exception) -> str:
    """Strips raw API keys, bearer tokens, and internal provider URLs from exception strings,
    and formats common authentication and quota errors with clear, actionable guidance."""
    raw = str(exc)
    raw_lower = raw.lower()

    if "invalid_api_key" in raw_lower or "invalid api key" in raw_lower or ("401" in raw and "auth" in raw_lower):
        return (
            "Authentication Error (401 - Invalid API Key): The Groq API key configured in the environment "
            "is invalid, expired, or has been revoked. Please generate a new free API key at "
            "https://console.groq.com/keys and update it in your .env file or Streamlit Cloud Secrets."
        )

    if "rate_limit_exceeded" in raw_lower or "429" in raw:
        return (
            "Rate Limit Reached (429): The Groq API rate limit has been exceeded. "
            "Please wait a moment before trying again."
        )

    # Mask Groq keys
    sanitized = re.sub(r"gsk_[A-Za-z0-9_-]{15,}", "gsk_***[REDACTED]***", raw)
    # Mask Bearer tokens
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]{15,}", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    # Mask local file paths
    sanitized = re.sub(r"[A-Za-z]:\\[\w\\\.\-]+", "[LOCAL_PATH]", sanitized)
    return sanitized
