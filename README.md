---
title: Surface Auto-Annotator
emoji: 🛣️
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
---

# Surface Auto-Annotator

A Streamlit + SegFormer app that auto-segments **road, walkway and bikepath**
paved surfaces. Each class is emitted as its **own polygon** (YOLO-seg +
CVAT), so the annotation is precise and per-class instead of a single merged
blob. Green vegetation and terrain are **not** annotated.

Optionally detects **cars / trucks / buses** with YOLOv8-seg and emits them as
a separate `car` class (id `3`) so they can overlap the surface polygon
(e.g. a car parked on the road, blocking part of the lane).

- **Model:** `nvidia/segformer-b3-finetuned-cityscapes-1024-1024` (Hugging Face,
  downloaded automatically on first run into `models/`).
- **Dual model (optional):** A second SegFormer-B0 model runs in parallel and
  their outputs are merged (intersection = both must agree, union = either
  detects). This improves accuracy by combining two different model capacities.

## What "more precise annotation" means here

Compared to a basic SegFormer pass, this app layers on several precision
improvements out of the box:

1. **Per-class polygons** — `surface_road`, `surface_walkway`, and
   `surface_bikepath` are three independent polygons instead of one merged
   blob. A walkway that runs beside a road is annotated as its own shape; a
   bike path that is painted inside the road is split out as `surface_bikepath`.
2. **Test-time augmentation (TTA, on by default)** — the model runs on the
   original image AND the horizontally-flipped image, then the per-pixel
   probabilities are averaged. Roads and walkways are left-right symmetric,
   so this is essentially a free precision win at ~2x runtime.
3. **Multi-scale inference (opt-in)** — runs the model at 512 / 768 / 1024 px
   and averages the probabilities. Stabilises boundary placement across
   variable-resolution photos. ~3x runtime.
4. **Edge-aware mask refinement** — bilateral filter (instead of plain
   Gaussian) keeps the boundary crisp, so the polygon follows the real
   surface edge instead of smudging it.
5. **Denser polygons** — smaller `approxPolyDP` epsilon so the polygon
   follows the real boundary closely (more vertices, tighter fit).
6. **Multi-polygon per class** — a road that splits around a pedestrian
   island or a walkway that fades out is preserved as multiple polygons
   instead of being merged into one.
7. **Bike path detection** — the road mask is searched for a red-painted or
   curb-bordered strip; when found, it's split out into `surface_bikepath`.
   When not, the road polygon stays whole.

## Classes

- `0 surface_road` — Cityscapes class `0 road` (paved surfaces including
  anything the model lumps as road).
- `1 surface_walkway` — Cityscapes class `1 sidewalk` (walkway).
- `2 surface_bikepath` — sub-region of the road mask detected by a
  red-paint + edge heuristic. Emitted only when confidently found.
- `3 car` *(optional, YOLOv8-seg)* — cars, trucks, buses. Each car polygon
  is clipped to the surface mask, so only the part of the car that overlaps
  the road / walkway is emitted.

**Black regions** (e.g. dark car hoods) are excluded from surface
annotations.

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

Toggle **"Detect cars (YOLOv8-seg)"** in the sidebar to also annotate cars
that overlap the surface. The first run downloads `yolov8n-seg.pt`.

In the **Precision** section of the sidebar, you can toggle TTA (on by
default) and multi-scale inference, plus the bike-path detector.

## Run the CLI (batch)

Place input images in `images/`, then:

```powershell
.venv\Scripts\activate
python annotate.py
```

### Outputs

| Path | Description |
|---|---|
| `labels/*.txt` | YOLO-seg polygons. Line `0 <poly>` = road, `1 <poly>` = walkway, `2 <poly>` = bikepath, `3 <poly>` = car (when car detection is on). |
| `preview/` | Images with class-colored polygon outlines and translucent fills (red=road, green=walkway, cyan=bikepath, blue=car). |
| `annotated images/` | Copies of the annotated previews. |
| `cvat_annotations.xml` | CVAT-importable polygon annotations (registers `surface_road`, `surface_walkway`, `surface_bikepath`, and `car` when car detection is on). |
| `summary.csv` | Per-image counts and areas. |
| `classes.txt` / `dataset.yaml` | YOLO dataset config, one line per class. |

## Import into CVAT

> ⚠️ Polygons reference the labels `surface_road`, `surface_walkway`,
> `surface_bikepath` (and `car` if enabled). The **labels must exist on the
> task**. The "Upload annotations" action (menu ⋮ on an existing task) does
> **not** create labels — it only matches to existing ones — so importing
> onto a task without those labels raises
> `Label 'surface_road' is not registered`.

**Recommended — single zip upload (auto-registers the labels):**

1. Build a zip containing the images + XML, with matching filenames:
   ```
   surface-import.zip
   ├── cvat_annotations.xml
   ├── upload_0001.png
   └── ...
   ```
2. CVAT → **Tasks** → **Create task** → **Upload dataset** → choose the zip →
   pick format **CVAT XML for images** → create.
   CVAT reads `<meta><labels>` from the XML and registers all the
   `surface_*` labels (and `car`) automatically.

The export now emits fully-typed labels:
`<label><name>surface_road</name><type>polygon</type><attributes/></label>`, so
the auto-created labels support polygons.

**Alternative — attach to an existing task:** create/register
`surface_road`, `surface_walkway`, `surface_bikepath` (and optionally `car`)
polygon labels on the task first, then use **⋮ → Upload annotations** with
`cvat_annotations.xml`.

## Project layout

```
surface-annotator/
├── app.py              # Streamlit web UI
├── annotate.py         # CLI batch runner
├── annotator_core.py   # SegFormer model + annotation logic (shared)
├── requirements.txt    # pinned dependencies
├── classes.txt         # generated: 0 surface_road, 1 surface_walkway, 2 surface_bikepath, 3 car
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

- The app now emits **per-class polygons** for the paved surface: `surface_road`,
  `surface_walkway`, and `surface_bikepath`. Vegetation/terrain are excluded.
- With **"Detect cars"** on, the app also runs YOLOv8-seg and emits any car
  polygon **clipped to the surface mask** — so the car's footprint on the
  road is labeled, but the part of the car silhouette that is not over the
  surface (e.g. sky, sidewalk, vegetation) is left out.
- `models/yolov8n-seg.pt` is the YOLOv8-seg checkpoint used for car
  detection (downloaded on first use).
- The Streamlit UI's "Annotate" button runs the full pipeline per image
  (load model once, annotate, export YOLO-seg label + preview + CVAT XML).
- Preview images show **class-colored polygon outlines + translucent
  fills**: red = road, green = walkway, cyan = bike path, blue = car.
- **Dual model mode** runs a second SegFormer-B0 in parallel and merges
  their outputs using the configured strategy (intersection/union).

## Precision tuning

All knobs are constants at the top of `annotator_core.py`:

| Constant | Default | Effect |
|---|---|---|
| `CLASS_PREFIX` | `surface_` | Default prefix prepended to the class names. |
| `MODEL_NAME` | `segformer-b3-...cityscapes` | Backbone size. `b0` = fastest, `b5` = most precise. |
| `SECOND_MODEL_NAME` | `segformer-b0-...cityscapes` | Second model for dual-mode (different capacity). |
| `DUAL_MODE` | `False` | Enable second model running in parallel. |
| `MERGE_STRATEGY` | `intersection` | How to combine two model outputs. |
| `TTA` | `True` | Test-time augmentation (horizontal-flip average). |
| `MULTI_SCALE` | `False` | Run inference at 512/768/1024 and average. |
| `BIKEPATH_DETECT` | `True` | Try to split a bike path out of the road mask. |
| `TILE_SIZE` / `TILE_OVERLAP` | `1024` / `256` | Overlapping-tile inference for huge images. |
| `CONF_THRESHOLD` | `0.40` | Min softmax P(surface). Higher = stricter. |
| `REFINE_BILATERAL_D/SIGMA_*` | `5 / 25 / 25` | Edge-aware mask smoothing (keeps boundaries crisp). |
| `REFINE_OPEN_KSIZE` / `REFINE_CLOSE_KSIZE` | `3` / `7` | Morphological noise removal. |
| `MIN_COMPONENT_PIXELS` / `RELATIVE_MIN_FRACTION` | `200` / `0.02` | Keep every connected surface above a relative size. |
| `POLY_BLUR_KSIZE` / `POLY_EPSILON_FRACTION` | `3` / `0.00025` | Mask boundary smoothing + polygon vertex density. |
| `YOLO_MODEL_PATH` | `yolov8n-seg.pt` | YOLOv8-seg checkpoint used for car detection. |
| `CAR_CONF_THRESHOLD` | `0.35` | Min YOLOv8-seg confidence to keep a car instance. |
| `MAX_CAR_INSTANCES` | `20` | Cap on number of cars emitted per image. |
| `CAR_COCO_IDS` | `{2, 5, 7}` | COCO class ids treated as "car" (car, bus, truck). |

Tiled inference costs time: a ~4000px photo is ~25 forward passes (b3 on CPU
can take a few minutes). Lower `MODEL_NAME` to `b1` or reduce `TILE_OVERLAP`
to speed up at a small precision cost.
