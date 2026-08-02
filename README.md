---
title: Surface Auto-Annotator
emoji: 🛣️
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.60.0
app_file: app.py
pinned: false
---

# Surface Auto-Annotator

A Streamlit + SegFormer app that auto-segments **road, walkway and bikepath**
paved surfaces and merges them into a single **`surface_surface`** class, producing
YOLO-seg-style polygon labels and a red polygon outline preview. Green vegetation and
terrain are **not** annotated.

Optionally detects **cars / trucks / buses** with YOLOv8-seg and emits them as a
separate `car` class (id `1`) so they can overlap the surface polygon (e.g. a car
parked on the road, blocking part of the lane).

- **Model:** `nvidia/segformer-b3-finetuned-cityscapes-1024-1024` (Hugging Face,
  downloaded automatically on first run into `models/`).
- **Dual model (optional):** A second SegFormer-B0 model runs in parallel and
  their outputs are merged (intersection = both must agree, union = either detects).
  This improves accuracy by combining two different model capacities.
- **Classes:**
  - `0 surface_surface` — merges Cityscapes classes `0 road` (bikeways/paved paths
    fall here too — Cityscapes has no separate bikepath class) and `1 sidewalk`
    (walkway).
  - `1 car` *(optional, YOLOv8-seg)* — cars, trucks, buses. Each car polygon is
    clipped to the surface mask, so only the part of the car that overlaps the
    road / walkway is emitted (the car "overlaps" the surface exactly where the
    car is on top of it).
- **Black regions** (e.g. dark car hoods) are excluded from surface annotations.

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/) (CPU wheel works fine)

## Install

```powershell
cd C:\Users\dhyan\surface-annotator
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the web UI

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Drag and drop JPG/PNG images onto the uploader, then click **Annotate**.
The first run downloads the SegFormer model and may take ~30s.

Toggle **"Detect cars (YOLOv8-seg)"** in the sidebar to also annotate cars that
overlap the surface (clipped to the road / walkway region so only the visible,
on-surface part of the car is emitted). The first run downloads `yolov8n-seg.pt`.

## Run the CLI (batch)

Place input images in `images/`, then:

```powershell
.venv\Scripts\activate
python annotate.py
```

### Outputs

| Path | Description |
|---|---|
| `labels/*.txt` | YOLO-seg polygons. Line `0 <polygon>` = surface; line `1 <polygon>` = car (when car detection is on). |
| `preview/` | Images with red polygon outline (surface) and blue polygon outline (cars, when on). |
| `annotated images/` | Copies of the annotated previews. |
| `cvat_annotations.xml` | CVAT-importable polygon annotations (auto-registers `surface_surface` and, when car detection is on, `car`). |
| `summary.csv` | Per-image `surface_polys` and `area_px`. |
| `classes.txt` / `dataset.yaml` | YOLO dataset config (`0: surface_surface`, plus `1: car` when car detection is on). |

## Import into CVAT

> ⚠️ Polygons reference the label `surface_surface`, so the **label must exist on the
> task**. The "Upload annotations" action (menu ⋮ on an existing task) does
> **not** create labels — it only matches to existing ones — so importing onto a
> task without a `surface_surface` label raises `Label 'surface_surface' is not registered`.

**Recommended — single zip upload (auto-registers the label):**

1. Build a zip containing the images + XML, with matching filenames:
   ```
   surface-import.zip
   ├── cvat_annotations.xml
   ├── upload_0001.png
   └── ...
   ```
2. CVAT → **Tasks** → **Create task** → **Upload dataset** → choose the zip →
   pick format **CVAT XML for images** → create.
   CVAT reads `<meta><labels>` from the XML and registers `surface_surface` automatically.

The export now emits a fully-typed label:
`<label><name>surface_surface</name><type>polygon</type><attributes/></label>`, so the
auto-created label supports polygons.

**Alternative — attach to an existing task:**
create/register a **surface_surface** polygon label on the task first (Project/Labels
schema), then use **⋮ → Upload annotations** with `cvat_annotations.xml`.

## Project layout

```
surface-annotator/
├── app.py              # Streamlit web UI
├── annotate.py         # CLI batch runner
├── annotator_core.py   # SegFormer model + annotation logic (shared)
├── requirements.txt    # pinned dependencies
├── classes.txt         # generated: `surface_surface`
├── dataset.yaml        # generated: YOLO dataset config
├── cvat_annotations.xml# generated: CVAT export
├── summary.csv         # generated: per-image report
├── images/             # input images (drop here or use uploader)
├── labels/             # generated YOLO-seg labels
├── preview/            # generated annotated previews
├── annotated images/   # generated final copies
└── models/             # SegFormer + HF cache
```

## Notes

- Only **surface_surface** (road + walkway + bikepath, paved only) is detected by
  default, merged into one polygon. Vegetation/terrain are excluded.
- With **"Detect cars"** on, the app also runs YOLOv8-seg and emits any car
  polygon **clipped to the surface mask** — so the car's footprint on the road is
  labeled, but the part of the car silhouette that is not over the surface (e.g.
  sky, sidewalk, vegetation) is left out.
- `models/yolov8n-seg.pt` is the YOLOv8-seg checkpoint used for car detection
  (downloaded on first use).
- The Streamlit UI's "Annotate" button runs the full pipeline per image
  (load model once, annotate, export YOLO-seg label + preview + CVAT XML).
- Preview images show **red polygon outlines** only — no green fill overlay.
- **Dual model mode** runs a second SegFormer-B0 in parallel and merges
  their outputs using the configured strategy (intersection/union).

## Precision tuning

All knobs are constants at the top of `annotator_core.py`:

| Constant | Default | Effect |
|---|---|---|
| `CLASS_PREFIX` | `surface_` | Default prefix prepended to the class name in all annotation outputs. Configurable via the UI's "Class name prefix" field. |
| `MODEL_NAME` | `segformer-b3-...cityscapes` | Backbone size. `b0` = fastest, `b5` = most precise. |
| `SECOND_MODEL_NAME` | `segformer-b0-...cityscapes` | Second model for dual-mode (different capacity). |
| `DUAL_MODE` | `False` | Enable second model running in parallel. |
| `MERGE_STRATEGY` | `intersection` | How to combine two model outputs: `intersection` = both must agree (higher precision), `union` = either detects (higher recall). |
| `TILE_SIZE` / `TILE_OVERLAP` | `1024` / `256` | Inference runs in overlapping patches, so high-res photos are **not** downsampled; overlap is averaged to hide seams. |
| `CONF_THRESHOLD` | `0.5` | Min softmax P(surface). Higher = stricter/tighter mask, lower = more recall. |
| `REFINE_OPEN_KSIZE` / `REFINE_CLOSE_KSIZE` | `5` / `7` | Morphological noise removal. Reduce if edges look over-smoothed. |
| `MIN_COMPONENT_PIXELS` / `RELATIVE_MIN_FRACTION` | `200` / `0.02` | Keeps every connected road segment above a relative size (long thin roads that split are no longer dropped). |
| `POLY_BLUR_KSIZE` / `POLY_EPSILON_FRACTION` | `5` / `0.0005` | Mask boundary smoothing + polygon vertex density. |
| `YOLO_MODEL_PATH` | `yolov8n-seg.pt` | YOLOv8-seg checkpoint used for car detection (placed in `models/`). |
| `CAR_CONF_THRESHOLD` | `0.35` | Min YOLOv8-seg confidence to keep a car instance. |
| `MAX_CAR_INSTANCES` | `20` | Cap on number of cars emitted per image. |
| `CAR_COCO_IDS` | `{2, 5, 7}` | COCO class ids treated as "car" (car, bus, truck). |

Tiled inference costs time: a ~4000px photo is ~25 forward passes (b3 on CPU
can take a few minutes). Lower `MODEL_NAME` to `b1` or reduce `TILE_OVERLAP`
to speed up at a small precision cost.
