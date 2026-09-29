"""
YOLO Perception Extractor
─────────────────────────
Uses ultralytics YOLOv8 to detect road users and produce a structured
spatial-context text summary for Condition B of the experiment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── COCO class-ID → human-readable label mapping (road-user subset) ─────
ROAD_USER_CLASSES: dict[int, str] = {
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Relative-area buckets → qualitative proximity labels ─────────────────
_PROXIMITY_THRESHOLDS: list[tuple[float, str]] = [
    (0.25, "very large (critical proximity)"),
    (0.10, "large (close proximity)"),
    (0.03, "medium"),
    (0.00, "small (distant)"),
]

# ── Colour palette for bounding-box overlays (BGR) ───────────────────────
_BOX_COLOURS: dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),      # pedestrian – red
    1: (0, 200, 255),    # cyclist    – orange
    2: (255, 200, 0),    # car        – cyan
    3: (255, 0, 200),    # motorcycle – magenta
    5: (0, 255, 0),      # bus        – green
    7: (200, 200, 0),    # truck      – teal
}


@dataclass
class DetectedObject:
    """A single detected road user with spatial metadata."""

    class_id: int
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    relative_area: float          # bbox area / image area
    proximity_label: str          # qualitative proximity bucket
    horizontal_position: str      # "left side", "center", "right side"
    depth_position: str           # "foreground / near vehicle", "midground", "background"


@dataclass
class PerceptionResult:
    """Full perception output for one image."""

    image_path: str
    image_width: int
    image_height: int
    detections: List[DetectedObject] = field(default_factory=list)

    # ── Summarisation ────────────────────────────────────────────────
    def to_text_summary(self) -> str:
        """
        Build a human-readable spatial-context summary suitable for
        injection into the Condition B VLM prompt.
        """
        if not self.detections:
            return "Detected Road Users: None."

        # Count per class
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.label] = counts.get(det.label, 0) + 1

        header_parts = [f"{cnt} {label}{'s' if cnt > 1 else ''}"
                        for label, cnt in counts.items()]
        header = "Detected Road Users: " + ", ".join(header_parts) + "."

        lines = [header]
        for det in self.detections:
            lines.append(
                f"  - 1 {det.label} ({det.proximity_label}) located near "
                f"{det.horizontal_position} ({det.depth_position}) "
                f"[confidence: {det.confidence:.2f}]."
            )
        return "\n".join(lines)


class YOLOPerceptionExtractor:
    """
    Wraps YOLOv8 to detect road users and produce structured metadata.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier (e.g. ``"yolov8n.pt"``).
    confidence_threshold : float
        Minimum detection confidence to keep.
    device : str | None
        Force device (``"cpu"``, ``"cuda:0"``).  ``None`` = auto.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = confidence_threshold
        self.device = device

    # ── Core pipeline ────────────────────────────────────────────────

    def extract(self, image_path: str | Path) -> PerceptionResult:
        """
        Run YOLO on *image_path* and return a ``PerceptionResult``
        containing only road-user detections with spatial metadata.
        """
        image_path = str(image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[DetectedObject] = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in ROAD_USER_CLASSES:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                box_area = (x2 - x1) * (y2 - y1)
                rel_area = box_area / img_area

                detections.append(DetectedObject(
                    class_id=cls_id,
                    label=ROAD_USER_CLASSES[cls_id],
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    relative_area=rel_area,
                    proximity_label=self._proximity_label(rel_area),
                    horizontal_position=self._horizontal_position(x1, x2, img_w),
                    depth_position=self._depth_position(y1, y2, img_h),
                ))

        return PerceptionResult(
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
            detections=detections,
        )

    # ── Annotated image I/O ──────────────────────────────────────────

    def save_annotated_image(
        self,
        image_path: str | Path,
        perception: PerceptionResult,
        output_path: str | Path,
    ) -> str:
        """
        Draw bounding boxes with labels on *image_path* and save to
        *output_path*.  Returns the absolute output path.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        for det in perception.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            colour = _BOX_COLOURS.get(det.class_id, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

            label_text = (
                f"{det.label} {det.confidence:.2f} | "
                f"{det.proximity_label}"
            )
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )
            cv2.rectangle(
                img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), colour, -1,
            )
            cv2.putText(
                img, label_text, (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return str(Path(output_path).resolve())

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _proximity_label(relative_area: float) -> str:
        for threshold, label in _PROXIMITY_THRESHOLDS:
            if relative_area >= threshold:
                return label
        return "small (distant)"

    @staticmethod
    def _horizontal_position(x1: float, x2: float, img_w: int) -> str:
        cx = (x1 + x2) / 2
        third = img_w / 3
        if cx < third:
            return "left side"
        elif cx < 2 * third:
            return "center"
        else:
            return "right side"

    @staticmethod
    def _depth_position(y1: float, y2: float, img_h: int) -> str:
        """
        Use vertical placement as a monocular depth proxy:
        objects nearer the bottom of the frame are closer to the ego vehicle.
        """
        cy = (y1 + y2) / 2
        if cy > 0.70 * img_h:
            return "foreground / near vehicle"
        elif cy > 0.40 * img_h:
            return "midground"
        else:
            return "background"
