"""guardrails.py - Task 10: PII masking, prompt-injection detection, groundedness check."""
import re

# Phone: 10 digits (6-9 se shuru), optional +91 / space / dash
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Card last-4: "card ending 4321", "last 4 digits 4321" jaise phrases ke baad ke 4 digits
CARD_RE = re.compile(
    r"(?i)((?:card[^.\d]{0,25}|last\s*4\s*digits?[^.\d]{0,15}|ending\s*(?:in|with)?\s*))(\d{4})(?!\d)"
)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|any\s+)?(previous|prior|above|your)\s+(instructions|rules)",
    r"(reveal|show|print|tell me)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+",
    r"forget\s+(everything|all)\s+(you|above|previous)",
    r"developer\s+mode",
    r"jailbreak",
    r"pretend\s+(to\s+be|you\s+are)",
]
_INJ = [re.compile(p, re.I) for p in INJECTION_PATTERNS]


def mask_pii(text):
    """Returns (masked_text, list of PII types found)."""
    found = []

    def phone_sub(m):
        found.append("phone")
        return "[PHONE_MASKED]"

    def card_sub(m):
        found.append("card_last4")
        return m.group(1) + "[CARD_LAST4_MASKED]"

    text = PHONE_RE.sub(phone_sub, text)
    text = CARD_RE.sub(card_sub, text)
    return text, found


def detect_injection(text):
    """Returns (True/False, matched pattern)."""
    for rx in _INJ:
        if rx.search(text):
            return True, rx.pattern
    return False, None


def check_groundedness(top_similarity, threshold, answer, chunks):
    """Output-side check: (ok, reason). Refuse agar context support na kare."""
    if top_similarity < threshold:
        return False, f"top similarity {top_similarity} < threshold {threshold}"
    body = answer.replace("According to our policy:", "").strip()
    context = " ".join(c["text"] for c in chunks)
    for sent in re.split(r"(?<=[.!?])\s+", body):
        if sent and sent not in context:
            return False, "answer sentence not found in retrieved context"
    return True, "ok"