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
    "You are an expert traffic safety analyst evaluating situational complexity and "
    "driver attention requirements for an autonomous vehicle supervisor."
)

FACTORS_RUBRIC: str = (
    "Assess situational complexity by evaluating these 5 core factors from 0 to 10:\n"
    "  1. traffic_density: Density, proximity, and behavior of surrounding vehicles.\n"
    "  2. vulnerable_road_users: Pedestrians, cyclists, or animals near the roadway.\n"
    "  3. conflict_proximity: Any road user or obstacle close to or crossing the ego vehicle path.\n"
    "  4. road_complexity: Intersections, roundabouts, lane merges, tracks, construction.\n"
    "  5. visibility: Lighting conditions, weather, sun glare, or sight-line occlusions.\n\n"
    "Scoring rubric (0 to 10):\n"
    "  • 0–1: Absent or irrelevant (clear/empty)\n"
    "  • 2–3: Minor, far away, routine monitoring\n"
    "  • 4–6: Noticeable, active vigilance needed\n"
    "  • 7–8: Close or active, likely to matter\n"
    "  • 9–10: Immediate conflict or critical hazard\n\n"
    "Use the full range; a typical quiet street should score mostly 0–3.\n"
    "For EACH factor, give exactly one short sentence explaining your reason BEFORE its score.\n\n"
    "Respond strictly as JSON in this exact structure:\n"
    "{\n"
    '  "traffic_density": {"reason": "<1 short sentence>", "score": <0-10>},\n'
    '  "vulnerable_road_users": {"reason": "<1 short sentence>", "score": <0-10>},\n'
    '  "conflict_proximity": {"reason": "<1 short sentence>", "score": <0-10>},\n'
    '  "road_complexity": {"reason": "<1 short sentence>", "score": <0-10>},\n'
    '  "visibility": {"reason": "<1 short sentence>", "score": <0-10>}\n'
    "}"
)

# ── Condition A — Image-Only ─────────────────────────────────────────────

def build_condition_a_prompt() -> str:
    """
    Return the user-turn prompt for **Condition A** (image-only).
    """
    return (
        "Analyze this traffic scene image from the perspective of an automated "
        "vehicle supervisor.\n\n"
        f"{FACTORS_RUBRIC}"
    )

# ── Condition B — Image + YOLO Spatial Metadata ─────────────────────────

def build_condition_b_prompt(yolo_summary: str) -> str:
    """
    Return the user-turn prompt for **Condition B** (image + YOLO metadata).
    """
    return (
        "Analyze this traffic scene image from the perspective of an automated "
        "vehicle supervisor.\n\n"
        "Here is explicit object detection data extracted from the scene:\n"
        f"{yolo_summary}\n\n"
        "Combined with the visual image, evaluate the scene:\n\n"
        f"{FACTORS_RUBRIC}"
    )