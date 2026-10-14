"""LLM word emphasis for burned-in captions (opt-in `caption_emphasis: llm`).

`caption.align.pick_emphasis_words` is the free, deterministic default. This module is the
model-assisted alternative: it asks the same OpenAI-compatible chat endpoint the caption pass
already uses which words in the transcript deserve highlighting.

The validation is the point. A model can answer with a word that was never said, a two-word
phrase, an essay or a numbered list, so every candidate is lowercased, stripped of surrounding
punctuation, dropped unless it occurs in the transcript as a whole word, and capped. A
hallucinating model can therefore only ever *choose among* words that were really spoken, and
any failure degrades to the heuristic rather than failing the render.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Sequence

import httpx

from clippy.caption.align import DEFAULT_EMPHASIS_LIMIT, Cue, cue_text

logger = logging.getLogger(__name__)

EMPHASIS_SYSTEM_PROMPT = """You pick which words deserve emphasis in a short-form video caption.
Choose at most {limit} words: the ones carrying the reaction, the joke, the number or the
punchline. Reply with one word per line, lowercase, exactly as it appears in the transcript,
with no punctuation, numbering or explanation. Reply with nothing at all if nothing deserves
emphasis."""

# The ASS writer matches emphasis with `word.strip().strip(".,!?\"'()[]").lower()`, so the same
# normalisation is applied here. A form the writer cannot match is worse than no emphasis.
_PUNCTUATION = ".,!?\"'()[]"
_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z']+")


def word_form(text: str) -> str:
    """Lowercase, unpunctuated form of a word, exactly as the ASS writer matches it."""
    return text.strip().strip(_PUNCTUATION).lower()


def transcript_word_forms(cues: Sequence[Cue]) -> set[str]:
    """Every word actually spoken, in the form the ASS writer matches."""
    return {form for cue in cues for word in cue.words if (form := word_form(word.text))}


def parse_emphasis_response(content: str) -> list[str]:
    """
    Split a model reply into ordered candidate words.

    The prompt asks for one word per line, but models add bullets, commas, quotes or a
    sentence of preamble, so tokens are separated on any non-word character. Order is
    preserved because the model's own ranking is a usable tie-break for the cap.
    """
    return [form for raw in _TOKEN_SPLIT.split(content) if (form := word_form(raw))]


def validate_emphasis_words(
    candidates: Iterable[str],
    *,
    allowed: set[str],
    limit: int = DEFAULT_EMPHASIS_LIMIT,
) -> set[str]:
    """
    Keep the first `limit` distinct candidates that really occur in the transcript.

    `allowed` comes from `transcript_word_forms`, so anything invented, misspelled,
    multi-word or merely echoed from the prompt is dropped here.
    """
    kept: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        form = word_form(raw)
        if not form or form in seen or form not in allowed:
            continue
        seen.add(form)
        kept.append(form)
        if len(kept) >= limit:
            break
    return set(kept)


def request_emphasis_words(
    cues: Sequence[Cue],
    *,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    limit: int = DEFAULT_EMPHASIS_LIMIT,
    timeout: float = 30.0,
) -> set[str]:
    """Ask the model which words to emphasise. Raises on HTTP errors."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": EMPHASIS_SYSTEM_PROMPT.format(limit=limit),
                },
                {"role": "user", "content": json.dumps({"transcript": cue_text(cues)})},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return set()
    content = choices[0].get("message", {}).get("content") or ""
    return validate_emphasis_words(
        parse_emphasis_response(str(content)),
        allowed=transcript_word_forms(cues),
        limit=limit,
    )


def pick_emphasis_words_llm(
    cues: Sequence[Cue],
    *,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    limit: int = DEFAULT_EMPHASIS_LIMIT,
    timeout: float = 30.0,
) -> tuple[set[str], str | None]:
    """
    Never raises: returns ``(words, failure)``.

    `failure` is set - and the caller falls back to the heuristic scorer - when there is no
    usable API key, the request fails, or the model returned nothing that occurs in the
    transcript. An empty set with `failure = None` cannot happen, so "no emphasis" is always
    explainable from `plan.json`.
    """
    key = (api_key or "").strip()
    if not key:
        return set(), "no API key"
    try:
        words = request_emphasis_words(
            cues,
            api_key=key,
            base_url=base_url,
            model=model,
            limit=limit,
            timeout=timeout,
        )
    except Exception as exc:  # emphasis must never fail a render
        logger.exception("LLM emphasis request failed")
        return set(), f"llm emphasis request failed: {exc}"
    if not words:
        return set(), "llm emphasis returned no word from the transcript"
    return words, None
