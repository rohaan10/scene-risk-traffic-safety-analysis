"""
VLM Client – Multi-Provider Wrapper
─────────────────────────────────────
Stateless client that sends an image (+ optional YOLO text) to a
Vision-Language Model and parses the structured two-line response.

Supported back-ends (selected via ``provider`` argument):
  • ``"gemini"``  – Google Gemini API  (``google-genai``)
  • ``"openai"``  – OpenAI GPT-4o / GPT-4 Vision  (``openai``)

The client is deliberately stateless: each call is independent with
no conversation memory, ensuring fair experimental conditions.
"""

from __future__ import annotations

import base64
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

_CONFIDENCE_RE = re.compile(r"[Cc]onfidence\s*:\s*(\d{1,3})")
_MEANING_RE = re.compile(r"[Mm]eaning\s*:\s*(.+)", re.DOTALL)


def parse_vlm_response(raw: str) -> VLMResponse:
    """
    Extract ``confidence_score`` and ``meaning`` from the VLM's
    two-line output.  Falls back gracefully on malformed replies.
    """
    score_match = _CONFIDENCE_RE.search(raw)
    meaning_match = _MEANING_RE.search(raw)

    score = int(score_match.group(1)) if score_match else -1
    score = max(0, min(score, 100)) if score >= 0 else -1

    meaning = meaning_match.group(1).strip() if meaning_match else raw.strip()

    return VLMResponse(
        confidence_score=score,
        meaning=meaning,
        raw_text=raw,
    )


# ── Helper: encode image to base64 data-URI ─────────────────────────────

def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
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
        ``"gemini"`` or ``"openai"``.
    model : str | None
        Model identifier (e.g. ``"gemini-3.8-flash"`` or ``"gpt-4o"``).
        Defaults are chosen per provider if omitted.
    api_key : str | None
        Explicit API key.  Falls back to the ``GEMINI_API_KEY`` or
        ``OPENAI_API_KEY`` environment variable.
    """

    _DEFAULTS = {
        "gemini": "gemini-3.8-flash",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model or self._DEFAULTS.get(self.provider, "gemini-3.8-flash")
        self.api_key = api_key

        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider!r}")

    # ── Provider initialisers ────────────────────────────────────────

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
                if self.provider == "gemini":
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
        data_uri = _image_to_base64(image_path)

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
