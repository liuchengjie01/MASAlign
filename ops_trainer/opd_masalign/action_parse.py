"""Split an AutoGLM-style response into (thinking, action). No third-party
imports on purpose: extract_critical_steps.py (data prep) and client.py
(inference) both need this, and only the latter has the openai package.
"""
from __future__ import annotations

ACTION_CALL_PREFIXES = ("do(action=", "finish(message=")


def find_matching_close_paren(content: str, open_idx: int) -> int | None:
    """content[open_idx] must be '('. Returns the index of the matching ')',
    tracking paren depth and skipping any '(' / ')' inside a quoted string
    argument (e.g. finish(message="... (note) ...")) -- a plain non-greedy
    regex up to the first ')' cuts such a call off at the inner ')'.
    Confirmed against real data: one of 250 real autoglm-phone-9b historical
    responses has a finish(message=...) whose message contains a literal
    parenthetical.

    Returns None if unbalanced (e.g. the call was truncated mid-argument).
    """
    depth = 0
    quote = None
    i = open_idx
    while i < len(content):
        ch = content[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def find_action_calls(content: str) -> list[tuple[int, int]]:
    """Every do(action=...)/finish(message=...) call in content as
    (start, end) spans (end exclusive of the closing ')'), ordered by start.
    """
    spans = []
    for prefix in ACTION_CALL_PREFIXES:
        start = 0
        while True:
            idx = content.find(prefix, start)
            if idx == -1:
                break
            close_idx = find_matching_close_paren(content, idx + prefix.index("("))
            if close_idx is not None:
                spans.append((idx, close_idx + 1))
            start = idx + len(prefix)
    spans.sort()
    return spans


def parse_response(content: str) -> tuple[str, str, bool]:
    """Returns (thinking, action, well_formed). See client.ModelClient._parse_response
    for why the split is anchored on the call that TERMINATES the response
    rather than on the first/last substring occurrence.
    """
    if "<answer>" in content:
        idx = content.rfind("<answer>")
        thinking = content[:idx].replace("<think>", "").replace("</think>", "").strip()
        action = content[idx + len("<answer>"):].replace("</answer>", "").strip()
        return thinking, action, True

    spans = find_action_calls(content)
    if not spans:
        return "", content, False

    for start, end in reversed(spans):
        tail = content[end:].strip()
        if tail.startswith("</answer>"):
            tail = tail[len("</answer>"):].strip()
        if not tail:
            return content[:start].strip(), content[start:end], True

    start, end = spans[-1]
    return content[:start].strip(), content[start:end], False


def parse_response_markdown_headers(content: str) -> tuple[str, str, bool]:
    """Split a GUI-Owl/Mobile-Agent-V3-style response into (thinking, action, well_formed).
    This model's `action_response` has no special tokens, just markdown-ish section headers:
        ### Thought ###
        <reasoning>

        ### Action ###
        {"action": "click", "coordinate": [x, y]}

        ### Description ###
        <one-line human description, not used here>
    well_formed is False when "### Action ###" is missing (response truncated or malformed) --
    the caller (extract_critical_steps.py) only trusts the parsed action when it also matches
    the step's own logged `action` field, same cross-check as parse_response above.
    """
    thought_marker, action_marker, desc_marker = "### Thought ###", "### Action ###", "### Description ###"
    if action_marker not in content:
        return "", content.strip(), False
    before, after = content.split(action_marker, 1)
    thinking = before.replace(thought_marker, "").strip()
    action = after.split(desc_marker, 1)[0].strip()
    return thinking, action, True
