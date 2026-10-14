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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.reasoning.prompts import (
    SYSTEM_PROMPT,
    build_condition_a_prompt,
    build_condition_b_prompt,
)

# ── 5-Factor Complexity Architecture ─────────────────────────────────────

FACTORS = [
    "traffic_density",
    "vulnerable_road_users",
    "conflict_proximity",
    "road_complexity",
    "visibility",
]

WEIGHTS = {
    "traffic_density": 0.20,
    "vulnerable_road_users": 0.25,
    "conflict_proximity": 0.30,
    "road_complexity": 0.15,
    "visibility": 0.10,
}


def factor_schema() -> dict:
    """JSON schema for factor-based situational complexity evaluation."""
    return {
        "type": "object",
        "properties": {
            f: {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": f"At most 1 concise sentence explaining {f}.",
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": f"Score for {f} from 0 to 10.",
                    },
                },
                "required": ["reason", "score"],
            }
            for f in FACTORS
        },
        "required": FACTORS,
    }


def _normalize_factor_score(raw_value: Any) -> int:
    """Normalize a factor score to an integer in [0, 10]."""
    try:
        val = float(raw_value)
    except (ValueError, TypeError):
        return 0
    # If the model emitted a 0-1 float (e.g. 0.8), scale to 0-10
    if 0.0 < val < 1.0:
        val = val * 10
    score = int(round(val))
    return max(0, min(score, 10))


def aggregate_factor_scores(factors_data: dict[str, dict[str, Any]]) -> int:
    """
    Aggregate 5 factor scores (0-10) into a continuous 0-100 attention score
    using a max-aware blend of weighted mean (60%) and peak single factor (40%).
    """
    scores = {}
    for f in FACTORS:
        sub = factors_data.get(f, {})
        if isinstance(sub, dict):
            scores[f] = _normalize_factor_score(sub.get("score", 0))
        else:
            scores[f] = _normalize_factor_score(sub)

    wmean = sum(WEIGHTS[f] * scores[f] for f in FACTORS)  # 0-10
    peak = max(scores.values()) if scores else 0           # 0-10
    final_score = round(10 * (0.6 * wmean + 0.4 * peak))   # 0-100
    return max(0, min(final_score, 100))


def determine_advisory_state(score: Optional[int | float]) -> str:
    """
    Map continuous 0-100 attention score to driver advisory state:
      - Incomplete (None, NaN, or <0)
      - Relax (<25)
      - Stay Alert (25-64)
      - Take Control (>=65)
    """
    if score is None or (isinstance(score, float) and (score < 0 or score != score)):
        return "Incomplete"
    if score < 0:
        return "Incomplete"
    elif score < 25:
        return "Relax"
    elif score < 65:
        return "Stay Alert"
    else:
        return "Take Control"


@dataclass
class VLMResponse:
    """Parsed output from the VLM."""

    confidence_score: Optional[int]                 # 0–100 aggregated driver attention / risk score (None if Incomplete)
    meaning: str                                    # qualitative explanation / synthesized reasoning
    raw_text: str                                   # full model reply (for debugging)
    factors: dict[str, dict[str, Any]] = field(     # sub-scores & reasons per factor
        default_factory=dict
    )
    advisory_state: str = "Stay Alert"             # Relax, Stay Alert, Take Control, Incomplete

    @property
    def attention_score(self) -> Optional[int]:
        """Alias for confidence_score representing driver attention required."""
        return self.confidence_score


# ── Response parser ──────────────────────────────────────────────────────

_SCORE_RE = re.compile(
    r"\*{0,2}(?:[Aa]ttention(?:\s*[Ss]core)?|[Ss]core|[Rr]isk\s*[Ss]core|[Cc]onfidence)\*{0,2}\s*:\s*\*{0,2}\s*(\d{1,3}(?:\.\d+)?)"
)
_CONFIDENCE_RE = _SCORE_RE

_MEANING_RE = re.compile(
    r"\*{0,2}(?:[Mm]eaning|[Rr]eason|[Ee]xplanation)\*{0,2}\s*:\s*\*{0,2}\s*(.*?)(?=\n*\*{0,2}(?:[Aa]ttention(?:\s*[Ss]core)?|[Ss]core|[Rr]isk\s*[Ss]core|[Cc]onfidence)\*{0,2}\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)


def _normalize_score(raw_value: str | float | int) -> Optional[int]:
    """Convert a flat raw score value to an integer in [0, 100], or None if invalid."""
    try:
        val = float(raw_value)
    except (ValueError, TypeError):
        return None

    if isinstance(raw_value, float) or (isinstance(raw_value, str) and "." in raw_value):
        if 0.0 <= val <= 1.0:
            val = val * 100
    elif 0.0 < val < 1.0:
        val = val * 100

    score = int(round(val))
    if score < 0 or score > 100:
        return None
    return score


def parse_vlm_response(raw: str) -> VLMResponse:
    """
    Extract factor sub-scores, aggregated score, and meaning from VLM output.

    Parsing strategy:
      1. JSON Factor Parsing  – extracts 5 factors {"reason": ..., "score": ...}
         and calculates max-aware aggregate.
      2. JSON Flat Parsing    – handles legacy flat responses {"attention": ..., "meaning": ...}.
      3. Regex Factor Parsing – extracts factors from malformed or truncated text.
      4. Regex Flat Fallback  – matches single Attention: / Meaning: lines.
    """
    clean = raw.strip()

    # ── 1. Try JSON parsing (strip markdown code fences if present) ───
    json_text = clean
    if "```" in json_text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            # A. Factor-based JSON
            factor_keys_present = [f for f in FACTORS if f in data]
            if factor_keys_present:
                factors_dict: dict[str, dict[str, Any]] = {}
                for f in FACTORS:
                    if f in data:
                        sub = data[f]
                        if isinstance(sub, dict):
                            f_score = _normalize_factor_score(sub.get("score", 0))
                            f_reason = str(sub.get("reason", "")).strip()
                        else:
                            f_score = _normalize_factor_score(sub)
                            f_reason = str(sub).strip()
                        factors_dict[f] = {"score": f_score, "reason": f_reason}

                # Only aggregate if all 5 factors are present; flag partial parses
                if len(factor_keys_present) == len(FACTORS):
                    agg_score = aggregate_factor_scores(factors_dict)
                    advisory = determine_advisory_state(agg_score)

                    # Synthesize meaning from active/notable factors
                    active = [
                        f"{f.replace('_', ' ').capitalize()} ({factors_dict[f]['score']}/10): {factors_dict[f]['reason']}"
                        for f in FACTORS
                        if factors_dict[f]["score"] >= 2 and factors_dict[f]["reason"]
                    ]
                    if not active:
                        active = [
                            f"{f.replace('_', ' ').capitalize()}: {factors_dict[f]['reason']}"
                            for f in FACTORS
                            if factors_dict[f]["reason"]
                        ]
                    meaning_text = "; ".join(active) if active else "Routine road conditions, no immediate hazards."

                    return VLMResponse(
                        confidence_score=agg_score,
                        meaning=meaning_text,
                        raw_text=raw,
                        factors=factors_dict,
                        advisory_state=advisory,
                    )
                else:
                    print(
                        f"    ⚠️ Incomplete factor parse ({len(factor_keys_present)}/{len(FACTORS)} factors recovered). "
                        f"Flagging as Incomplete (NaN)."
                    )
                    return VLMResponse(
                        confidence_score=None,
                        meaning=f"Incomplete factor parse ({len(factor_keys_present)}/{len(FACTORS)} factors recovered).",
                        raw_text=raw,
                        factors=factors_dict,
                        advisory_state="Incomplete",
                    )

            # B. Legacy Flat JSON
            score_val = -1
            for k in (
                "attention_score", "Attention_Score", "attention", "Attention",
                "Confidence", "confidence", "Score", "score", "Risk_Score", "risk_score",
            ):
                if k in data:
                    score_val = data[k]
                    break

            meaning = ""
            for k in (
                "meaning", "Meaning", "reason", "Reason", "explanation", "Explanation",
            ):
                if k in data:
                    meaning = data[k]
                    break

            score = _normalize_score(score_val)
            return VLMResponse(
                confidence_score=score,
                meaning=str(meaning).strip(),
                raw_text=raw,
                factors={},
                advisory_state=determine_advisory_state(score),
            )
    except (json.JSONDecodeError, AttributeError):
        pass  # fall through to regex

    # ── 2. Regex Factor Parsing (robust against truncated JSON) ───────
    factors_extracted: dict[str, dict[str, Any]] = {}
    for f in FACTORS:
        pattern = re.compile(
            rf'["\']?{f}["\']?\s*:\s*\{{?(?:[^{{}}]*?["\']?reason["\']?\s*:\s*["\']([^"\'\n]+)["\']?[,\s]*?)?["\']?score["\']?\s*:\s*(\d{{1,2}})',
            re.IGNORECASE,
        )
        m = pattern.search(clean)
        if m:
            r_str = (m.group(1) or "").strip()
            s_val = int(m.group(2))
            factors_extracted[f] = {"score": min(s_val, 10), "reason": r_str}

    if len(factors_extracted) == len(FACTORS):
        agg_score = aggregate_factor_scores(factors_extracted)
        advisory = determine_advisory_state(agg_score)
        active = [
            f"{f.replace('_', ' ').capitalize()}: {factors_extracted[f]['reason']}"
            for f in FACTORS
            if factors_extracted[f]["reason"]
        ]
        meaning_text = "; ".join(active) if active else "Scene evaluated via sub-scores."
        return VLMResponse(
            confidence_score=agg_score,
            meaning=meaning_text,
            raw_text=raw,
            factors=factors_extracted,
            advisory_state=advisory,
        )
    elif len(factors_extracted) >= 1:
        print(
            f"    ⚠️ Truncated VLM response: regex recovered only {len(factors_extracted)}/{len(FACTORS)} factors. "
            f"Flagging as Incomplete (NaN)."
        )
        return VLMResponse(
            confidence_score=None,
            meaning=f"Truncated response: {len(factors_extracted)}/{len(FACTORS)} factors recovered.",
            raw_text=raw,
            factors=factors_extracted,
            advisory_state="Incomplete",
        )

    # ── 3. Regex Flat Parsing (legacy format) ─────────────────────────
    score_matches = _SCORE_RE.findall(clean)
    meaning_match = _MEANING_RE.search(clean)

    score = _normalize_score(score_matches[-1]) if score_matches else None
    meaning = meaning_match.group(1).strip() if (meaning_match and meaning_match.group(1).strip()) else clean

    return VLMResponse(
        confidence_score=score,
        meaning=meaning,
        raw_text=raw,
        factors={},
        advisory_state=determine_advisory_state(score),
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
        "ollama": "gemma3:12b",
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
        self.model = model or self._DEFAULTS.get(self.provider, "gemma3:12b")
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
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"

        # JSON schema for 5-factor situational complexity evaluation
        response_schema = factor_schema()

        # Attempt structured schema constraint; fall back gracefully if unsupported
        try:
            response = self._ollama_client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                        "images": [image_path],
                    },
                ],
                format=response_schema,
                options={
                    "temperature": 0.2,
                    "num_predict": 500,
                },
                keep_alive="10m",
            )
            return response.message.content
        except Exception:
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
                    "num_predict": 500,
                },
                keep_alive="10m",
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
                max_output_tokens=500,
            ),
        )
        return response.text

    # ── OpenAI back-end ──────────────────────────────────────────────

    def _call_openai(self, image_path: str, user_prompt: str) -> str:
        data_uri = _image_to_data_uri(image_path)

        response = self._openai_client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=500,
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
