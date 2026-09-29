"""
Prompt Templates – Condition A & Condition B
─────────────────────────────────────────────
Stateless (memory-free) system and user prompts for the comparative
VLM risk-scoring experiment.

Both conditions instruct the VLM to output exactly two lines:
    Confidence: <0–100>
    Meaning: <brief qualitative explanation>
"""

from __future__ import annotations

# ── Shared system prompt (identical for both conditions) ─────────────────

SYSTEM_PROMPT: str = (
    "You are an expert traffic safety analyst working for an autonomous "
    "vehicle research programme.  You evaluate single traffic scene images "
    "and produce a numerical risk/danger score between 0 (completely safe, "
    "empty road) and 100 (extreme imminent danger, collision unavoidable).\n\n"
    "Guidelines:\n"
    "  • Assess overall scene risk from the perspective of an automated "
    "vehicle driver (ego vehicle).\n"
    "  • Consider pedestrian proximity, vehicle density, speed cues, road "
    "geometry, weather/visibility, and any unusual or hazardous behaviour.\n"
    "  • Be concise.  Do NOT produce any additional text beyond the two "
    "required output lines.\n\n"
    "Your output MUST follow this exact format (no extra lines):\n"
    "Confidence: <integer 0-100>\n"
    "Meaning: <one-sentence explanation of the primary hazard or reason "
    "for the score>"
)


# ── Condition A — Image-Only ─────────────────────────────────────────────

def build_condition_a_prompt() -> str:
    """
    Return the user-turn prompt for **Condition A** (image-only).

    The image itself is attached separately as a multimodal input;
    this function provides only the textual instruction.
    """
    return (
        "Analyze this traffic scene image from the perspective of an "
        "automated vehicle driver.  Rate the overall danger/risk score "
        "on a scale from 0 (no danger) to 100 (extreme danger).  Reply "
        "with exactly two lines:\n"
        "Confidence: <score>\n"
        "Meaning: <brief explanation>"
    )


# ── Condition B — Image + YOLO Spatial Metadata ─────────────────────────

def build_condition_b_prompt(yolo_summary: str) -> str:
    """
    Return the user-turn prompt for **Condition B** (image + YOLO metadata).

    Parameters
    ----------
    yolo_summary : str
        The spatial-context text produced by
        ``PerceptionResult.to_text_summary()``.
    """
    return (
        "Analyze this traffic scene image from the perspective of an "
        "automated vehicle driver.\n\n"
        "Here is explicit object detection data extracted from the scene:\n"
        f"{yolo_summary}\n\n"
        "Combined with the image, rate the overall danger/risk score on a "
        "scale from 0 (no danger) to 100 (extreme danger).  Reply with "
        "exactly two lines:\n"
        "Confidence: <score>\n"
        "Meaning: <brief explanation>"
    )
