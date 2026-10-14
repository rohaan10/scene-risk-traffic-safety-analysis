"""
Prompt Templates – Condition A & Condition B
─────────────────────────────────────────────
Stateless (memory-free) system and user prompts for the comparative
VLM driver-attention evaluation experiment.

Both conditions instruct the VLM to output exactly two lines:
    Confidence: <0–100>
    Meaning: <brief qualitative explanation>
"""

from __future__ import annotations

# ── Shared system prompt (identical for both conditions) ─────────────────

SYSTEM_PROMPT: str = (
    "You are an expert traffic safety analyst working for an autonomous "
    "vehicle research programme. You evaluate single traffic scene images "
    "and assess how much driver attention/vigilance is required, producing "
    "a numerical score between 0 (very little attention needed, calm/clear road) "
    "and 100 (full, immediate driver attention required, critical situation).\n\n"
    "Guidelines:\n"
    "  • Assess the required level of driver attention from the perspective "
    "of an automated vehicle driver or human supervisor (ego vehicle).\n"
    "  • Consider factors such as pedestrian proximity, vulnerable road users, "
    "vehicle density, unexpected obstacles, road geometry, speed cues, "
    "visibility, and any potential hazards requiring active monitoring or intervention.\n"
    "  • Be concise. Do NOT produce any additional text beyond the two "
    "required output lines.\n\n"
    "Your output MUST follow this exact format (no extra lines):\n"
    "Confidence: <integer 0-100>\n"
    "Meaning: <one-sentence explanation of why this level of driver attention is required>"
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
        "automated vehicle driver or supervisor. Rate how much attention "
        "the driver needs in this scene on a scale from 0 (very little attention "
        "required) to 100 (full, immediate attention required). Reply with "
        "exactly two lines:\n"
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
        "automated vehicle driver or supervisor.\n\n"
        "Here is explicit object detection data extracted from the scene:\n"
        f"{yolo_summary}\n\n"
        "Combined with the image, rate how much attention the driver needs "
        "in this scene on a scale from 0 (very little attention required) "
        "to 100 (full, immediate attention required). Reply with exactly "
        "two lines:\n"
        "Confidence: <score>\n"
        "Meaning: <brief explanation>"
    )