# Instructions — Traffic Scene Risk-Scoring Pipeline

## First-Time Setup

Follow these steps **once** when you clone the repository for the first time.

### 1. Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or higher |
| **uv** (recommended) | Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Linux/Mac) or `winget install astral-sh.uv` (Windows) |
| **pip** (alternative) | Already included with Python — use this if you prefer not to install `uv` |

### 2. Install dependencies

**With uv (recommended):**

```bash
uv sync
```

**With pip (alternative):**

```bash
pip install -r requirements.txt
```

### 3. Add traffic images

Place your traffic scene images (`.jpg` or `.png`) into the `data/raw_images/` folder:

```
data/
└── raw_images/
    ├── scene_001.jpg
    ├── scene_002.jpg
    └── ...
```

> **Note:** The `data/` folder is gitignored. Each team member must add their own images locally.

### 4. Set up API key (required for VLM scoring, skip if YOLO-only)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API Key"**
4. Copy the key

Then create a `.env` file in the project root:

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GEMINI_API_KEY=AIzaSy...your-key-here
```

> **Important:** A web subscription (Gemini Advanced, ChatGPT Plus) does **not** provide API access. You need an API key from Google AI Studio (free) or OpenAI Platform (paid).

> **Troubleshooting — "model not found" errors:** Google periodically retires older Gemini models. If you see a `404 NOT_FOUND` error mentioning the model is no longer available, update the default model in `src/reasoning/vlm_client.py` (look for `_DEFAULTS`) to the model name suggested in the error message. You can also override the model at runtime without editing code: `uv run python main.py --model gemini-3.8-flash`.

### 5. (Optional) Add ground truth scores

If you have human-annotated risk scores, create `data/ground_truth.csv`:

```csv
image_id,human_score
scene_001,45
scene_002,82
```

The `image_id` must match the image filename **without the extension** (e.g., `scene_001.jpg` → `scene_001`).

---

## Running the Pipeline

After completing first-time setup, use these commands to run experiments.

### YOLO-only (no API key needed)

Runs object detection on all images and saves annotated images with bounding boxes:

```bash
uv run python main.py --skip-vlm
```

Or with pip:

```bash
python main.py --skip-vlm
```

**Output:**
- `output/annotated_images/` — images with YOLO bounding box overlays
- `output/experiment_results.csv` — detection data (VLM scores will be `-1`)

### Full pipeline (YOLO + VLM)

Runs both Condition A (image-only) and Condition B (image + YOLO metadata) on every image:

```bash
uv run python main.py
```

Or with pip:

```bash
python main.py
```

**Output:**
- `output/annotated_images/` — images with YOLO bounding box overlays
- `output/experiment_results.csv` — full results with both VLM condition scores
- Console prints evaluation metrics if `data/ground_truth.csv` is present

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `gemini` | VLM provider: `gemini` or `openai` |
| `--model` | auto | Model name override (e.g., `gpt-4o`, `gemini-2.0-flash`) |
| `--yolo-model` | `yolo26n.pt` | YOLO model weights (auto-downloaded on first run) |
| `--yolo-conf` | `0.35` | YOLO confidence threshold |
| `--threshold` | `70` | Risk score cutoff for hazard classification metrics |
| `--images-dir` | `data/raw_images/` | Custom input images directory |
| `--skip-vlm` | off | Run YOLO only, skip VLM calls |

### Examples

```bash
# Use OpenAI instead of Gemini
uv run python main.py --provider openai

# Lower YOLO confidence to detect more objects
uv run python main.py --yolo-conf 0.25

# Change hazard classification threshold from 70 to 50
uv run python main.py --threshold 50

# Use a different YOLO model size (larger = more accurate, slower)
uv run python main.py --yolo-model yolo26s.pt
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/experiment_results.csv` | Per-image results: YOLO detections, Condition A score, Condition B score, ground truth |
| `output/annotated_images/` | Images with YOLO bounding boxes and proximity labels drawn on them |

> **Note:** The `output/` folder is gitignored. Results are generated locally and not committed to the repository.
