"""
VLM Client – Multi-Provider Wrapper
─────────────────────────────────────
Stateless client that sends an image (+ optional YOLO text) to a
Vision-Language Model and parses the structured two-line response.

Supported back-ends (selected via ``provider`` argument):
  • ``"ollama"``  – Local Ollama server (no API key required)
  • ``"gemini"``  – Google Gemini API  (``google-genai``)
  • ``"openai"``  – OpenAI GPT-4o / GPT-4 Vision  (``openai``)

The client is deliberately stateless: each call is independent with
no conversation memory, ensuring fair experimental conditions.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.reasoning.prompts import (
    SYSTEM_PROMPT,
    build_condition_a_prompt,
    build_condition_b_prompt,
)


@dataclass
class VLMResponse:
    """Parsed output from the VLM."""

    confidence_score: int        # 0–100 risk score
    meaning: str                 # qualitative explanation
    raw_text: str                # full model reply (for debugging)


# ── Response parser ──────────────────────────────────────────────────────

# Regex patterns – used as fallback when JSON parsing fails.
# These intentionally match both plain and markdown-bold formatted output
# (e.g. "**Confidence:** 45" or "Confidence: 0.83").
_CONFIDENCE_RE = re.compile(
    r"\*{0,2}[Cc]onfidence\*{0,2}\s*:\s*\*{0,2}\s*(\d{1,3}(?:\.\d+)?)"
)
_MEANING_RE = re.compile(
    r"\*{0,2}[Mm]eaning\*{0,2}\s*:\s*\*{0,2}\s*(.+)", re.DOTALL
)


def _normalize_score(raw_value: str | float | int) -> int:
    """
    Convert a raw score value to an integer in [0, 100].

    Handles:
      - Integers already in 0–100 range  (e.g. 45)
      - Floats in 0.0–1.0 range          (e.g. 0.83 → 83)
      - Float strings                     (e.g. "0.65" → 65)
    """
    try:
        val = float(raw_value)
    except (ValueError, TypeError):
        return -1

    # If the value is in 0–1 range (exclusive of 1), treat as a fraction
    if 0.0 <= val <= 1.0 and val != 0.0:
        val = val * 100

    score = int(round(val))
    return max(0, min(score, 100))


def parse_vlm_response(raw: str) -> VLMResponse:
    """
    Extract ``confidence_score`` and ``meaning`` from the VLM's output.

    Parsing strategy (in order of priority):
      1. JSON  – if the response is a valid JSON object.
      2. Regex – last match of ``Confidence:`` / ``Meaning:`` patterns
         (last match avoids capturing echoed YOLO confidence decimals).
      3. Fallback – score = -1, meaning = raw text.
    """
    clean = raw.strip()

    # ── 1. Try JSON parsing (strip markdown code fences if present) ───
    json_text = clean
    if json_text.startswith("```"):
        json_text = re.sub(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", r"\1",
            json_text, flags=re.DOTALL,
        ).strip()

    try:
        data = json.loads(json_text)
        score_val = data.get("Confidence", data.get("confidence", -1))
        meaning = data.get("Meaning", data.get("meaning", ""))
        score = _normalize_score(score_val)
        return VLMResponse(
            confidence_score=score,
            meaning=str(meaning).strip(),
            raw_text=raw,
        )
    except (json.JSONDecodeError, AttributeError):
        pass  # fall through to regex

    # ── 2. Regex parsing (take LAST match to skip YOLO echoes) ────────
    conf_matches = _CONFIDENCE_RE.findall(clean)
    meaning_match = _MEANING_RE.search(clean)

    # Use LAST confidence match — the model's answer typically comes
    # after any echoed YOLO detection confidence values.
    score = _normalize_score(conf_matches[-1]) if conf_matches else -1

    meaning = meaning_match.group(1).strip() if meaning_match else clean

    return VLMResponse(
        confidence_score=score,
        meaning=meaning,
        raw_text=raw,
    )


# ── Helper: encode image to base64 ──────────────────────────────────────

def _image_to_base64_raw(image_path: str) -> str:
    """Return raw base64 string (no data-URI prefix)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_to_data_uri(image_path: str) -> str:
    """Return full data-URI (for OpenAI)."""
    encoded = _image_to_base64_raw(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    return f"data:{mime};base64,{encoded}"


# ═════════════════════════════════════════════════════════════════════════
#  Public VLM Client
# ═════════════════════════════════════════════════════════════════════════

class VLMClient:
    """
    Unified VLM client for the risk-scoring experiment.

    Parameters
    ----------
    provider : str
        ``"ollama"``, ``"gemini"``, or ``"openai"``.
    model : str | None
        Model identifier.  Defaults are chosen per provider if omitted:
        ``"moondream"`` for Ollama, ``"gemini-3.8-flash"`` for Gemini,
        ``"gpt-4o"`` for OpenAI.
    api_key : str | None
        Explicit API key (not required for Ollama).
    ollama_host : str | None
        Ollama server URL.  Defaults to ``http://localhost:11434``.
    """

    _DEFAULTS = {
        "ollama": "qwen2-vl:7b",
        "gemini": "gemini-3.8-flash",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_host: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model or self._DEFAULTS.get(self.provider, "moondream")
        self.api_key = api_key
        self.ollama_host = ollama_host

        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider!r}")

    # ── Provider initialisers ────────────────────────────────────────

    def _init_ollama(self) -> None:
        import ollama

        host = self.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._ollama_client = ollama.Client(host=host)
        # Verify connectivity
        try:
            self._ollama_client.list()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {host}. "
                f"Make sure Ollama is running ('ollama serve') and the model "
                f"is pulled ('ollama pull {self.model}'). Error: {e}"
            )

    def _init_gemini(self) -> None:
        from google import genai

        key = self.api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "Set GEMINI_API_KEY env var or pass api_key= to VLMClient."
            )
        self._gemini_client = genai.Client(api_key=key)

    def _init_openai(self) -> None:
        from openai import OpenAI

        key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "Set OPENAI_API_KEY env var or pass api_key= to VLMClient."
            )
        self._openai_client = OpenAI(api_key=key)

    # ── Public API ───────────────────────────────────────────────────

    def score_image(
        self,
        image_path: str | Path,
        yolo_summary: Optional[str] = None,
        max_retries: int = 3,
        initial_backoff: float = 30.0,
    ) -> VLMResponse:
        """
        Send *image_path* (and optional *yolo_summary* for Condition B)
        to the VLM and return a parsed ``VLMResponse``.

        Automatically retries on rate-limit (429) and transient (503)
        errors with exponential backoff.

        Parameters
        ----------
        image_path : str | Path
            Path to the traffic scene image.
        yolo_summary : str | None
            If provided, Condition B prompt is used; otherwise Condition A.
        max_retries : int
            Maximum number of retry attempts on rate-limit / transient errors.
        initial_backoff : float
            Initial wait time in seconds before first retry (doubles each attempt).
        """
        image_path = str(image_path)

        if yolo_summary:
            user_prompt = build_condition_b_prompt(yolo_summary)
        else:
            user_prompt = build_condition_a_prompt()

        last_error: Exception | None = None
        for attempt in range(1 + max_retries):
            try:
                if self.provider == "ollama":
                    raw = self._call_ollama(image_path, user_prompt)
                elif self.provider == "gemini":
                    raw = self._call_gemini(image_path, user_prompt)
                else:
                    raw = self._call_openai(image_path, user_prompt)
                return parse_vlm_response(raw)
            except Exception as e:
                error_str = str(e)
                is_retryable = ("429" in error_str or "503" in error_str
                                or "RESOURCE_EXHAUSTED" in error_str
                                or "UNAVAILABLE" in error_str)
                if not is_retryable or attempt >= max_retries:
                    raise

                wait = initial_backoff * (2 ** attempt)
                print(f"    ⏳ Rate limited, retrying in {wait:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries}) …")
                time.sleep(wait)
                last_error = e

        raise last_error  # unreachable, but keeps type-checkers happy

    # ── Ollama back-end ──────────────────────────────────────────────

    def _call_ollama(self, image_path: str, user_prompt: str) -> str:
        """
        Call a local Ollama vision model.  Stateless: each call creates
        a fresh conversation with no history.
        """
        # Combine system prompt and user prompt into the message
        # (some smaller models don't fully respect system roles,
        #  so we prepend it to the user message for robustness)
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"

        response = self._ollama_client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                    "images": [image_path],
                },
            ],
            options={
                "temperature": 0.2,
                "num_predict": 256,
            },
        )
        return response.message.content

    # ── Gemini back-end ──────────────────────────────────────────────

    def _call_gemini(self, image_path: str, user_prompt: str) -> str:
        from google.genai import types

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        response = self._gemini_client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        types.Part.from_text(text=user_prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
        return response.text

    # ── OpenAI back-end ──────────────────────────────────────────────

    def _call_openai(self, image_path: str, user_prompt: str) -> str:
        data_uri = _image_to_data_uri(image_path)

        response = self._openai_client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "high"},
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content
