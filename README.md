# Traffic Scene Risk-Scoring Agent

A Python-based computer vision & multimodal AI system for traffic safety risk scoring, designed for a comparative experiment:

> **Does providing explicit object detection metadata (from YOLO) alongside traffic images improve a Vision Language Model's estimation of human-perceived traffic risk compared to image-only inputs?**

## Architecture

| Condition | Pipeline |
|-----------|----------|
| **A** (Image-Only) | Raw Image → VLM → Risk Score (0–100) + Explanation |
| **B** (Image + YOLO) | Raw Image + YOLO Spatial Context → VLM → Risk Score (0–100) + Explanation |

Both conditions are evaluated against human-annotated ground-truth risk scores.

## Repository Structure

```text
├── data/
│   ├── raw_images/           # Input traffic frames (.jpg, .png)
│   └── ground_truth.csv      # Human benchmark scores (image_id, human_score)
├── src/
│   ├── perception/
│   │   └── yolo_extractor.py # YOLOv8 detector & spatial metadata
│   ├── reasoning/
│   │   ├── prompts.py        # System & user prompt templates
│   │   └── vlm_client.py     # VLM API wrapper (Gemini / OpenAI)
│   ├── evaluation/
│   │   └── metrics.py        # Pearson r, Spearman ρ, F1, Precision, Recall
│   └── utils/
│       └── file_loader.py    # Directory walker & image utilities
├── output/
│   ├── annotated_images/     # YOLO bounding-box overlays
│   └── experiment_results.csv
├── main.py                   # Batch experiment runner
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
# Edit .env with your preferred editor
```

### 3. Add images

Place traffic scene images (`.jpg` / `.png`) into `data/raw_images/`.

### 4. (Optional) Add ground truth

Populate `data/ground_truth.csv`:

```csv
image_id,human_score
scene_001,45
scene_002,82
```

The `image_id` must match the filename stem (without extension).

### 5. Run the experiment

```bash
# Using Gemini (default)
python main.py --provider gemini

# Using OpenAI
python main.py --provider openai --model gpt-4o

# YOLO-only test (no VLM calls)
python main.py --skip-vlm
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `gemini` | VLM provider (`gemini` or `openai`) |
| `--model` | auto | Model name override |
| `--yolo-model` | `yolov8n.pt` | YOLO weights file |
| `--yolo-conf` | `0.35` | YOLO confidence threshold |
| `--threshold` | `70` | Risk binarisation threshold for classification metrics |
| `--images-dir` | `data/raw_images/` | Input images directory |
| `--skip-vlm` | `false` | Run YOLO only, skip VLM calls |

## Output

- **`output/experiment_results.csv`** — Per-image results with Condition A & B scores, YOLO summary, and ground truth
- **`output/annotated_images/`** — Images with YOLO bounding-box overlays and proximity labels
- **Console** — Pearson *r*, Spearman *ρ*, MAE, RMSE, Precision, Recall, F1 (if ground truth is available)

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Pearson *r* | Linear correlation with human scores |
| Spearman *ρ* | Rank-order correlation with human scores |
| MAE / RMSE | Absolute deviation from human scores |
| Precision / Recall / F1 | Binary hazard classification (score ≥ threshold) |