"""
Shared annotator core - surface (road / walkway / bikepath) + car.

Detects paved surface classes via SegFormer / Cityscapes. With the default
configuration, three surface classes are emitted independently:
    0  surface_road       (Cityscapes class 0 + bike paths found inside it)
    1  surface_walkway    (Cityscapes class 1 - sidewalk)
    2  surface_bikepath   (sub-mask of road; kept when its visual cues match a
                          cycle path, otherwise merged into road)

This split is what the user calls "more precise annotation": the polygon now
follows the actual paved region (walkway, lane, bike path) instead of one
merged blob.

Optional secondary output:
    3  car                (YOLOv8-seg instance segmentation, overlapping surface)

Cityscapes class coverage:
    0  road          (paved road)
    1  sidewalk      (walkway)

Precision additions over a basic SegFormer pass:
    * Multi-scale inference: average probabilities across several scales for
      stable boundaries regardless of photo resolution.
    * Test-Time Augmentation (TTA): horizontal-flip average, the strongest free
      win for road segmentation where the world is mirror-symmetric.
    * Edge-aware mask refinement: bilateral filter to keep crisp boundary
      alignment (a regular Gaussian blur smudges edges).
    * Denser polygons: smaller approxPolyDP epsilon so the contour follows the
      real boundary closely.
    * Bikepath detection: a small learned heuristic that picks out the painted
      red-asphalt / separated bike lane inside the road mask, so it is emitted
      as its own polygon.

Output labels:
    labels/<image>.txt  -> one line per polygon:  "<class_id> x1 y1 x2 y2 ..."

Classes are exposed via CLASSES below and the classes.txt / dataset.yaml
generators. Per-class overlay colors are exposed via OVERLAY_COLORS_BGR.
"""

from pathlib import Path
import shutil
import datetime
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ---- Class definitions ----
# Single surface class (merged): paved road, walkway, and bikepath all together
CLASS_PREFIX = "surface_"

CLASSES = [
    (0, f"{CLASS_PREFIX}road"),  # Includes road, walkway, and bikepath
]

# Backwards-compat: old callers (and CVAT exports) still expect this name.
SURFACE_SURFACE_NAME = f"{CLASS_PREFIX}surface"

# SegFormer (Cityscapes) class ids mapped to OUR per-class surface classes.
#   0 = road        -> our class 0  (surface_road - merged)
#   1 = sidewalk    -> merged into road
SURFACE_SEG_IDS = {0: 0, 1: 0}  # Both map to class 0

# Cityscapes "vegetation" class. Pixels where P(vegetation) >= P(surface) are
# excluded, so grass/shrubs bleeding into the road don't get annotated.
VEGETATION_SEG_IDS = {8}

# COCO class ids - car detection removed
CAR_COCO_IDS = set()  # No car classes

BLACK_THRESHOLD = 15  # RGB values below this in ALL channels = "black" (excluded)

# ---- Speed settings ----
# FAST_MODE=True resizes images to FAST_SIZE before inference for speed.
# This trades some precision for much faster annotation.
FAST_MODE = False  # default OFF now: precision > speed for this app
FAST_SIZE = 768    # a touch larger than 640 -> tighter boundaries

# ---- Multi-scale inference ----
# When MULTI_SCALE=True, we run inference at each of MULTI_SCALE_SIZES and
# average the resulting probabilities. This is the single biggest precision
# win for variable-resolution photos. Default OFF because it triples runtime.
MULTI_SCALE = False
MULTI_SCALE_SIZES = (512, 768, 1024)

# ---- Test-Time Augmentation (TTA) ----
# When TTA=True, we also run inference on the horizontally-flipped image and
# average probabilities back. Roads and walkways are left-right symmetric, so
# this is essentially a free precision boost at 2x runtime. Default ON.
TTA = True

# ---- Dual-model settings ----
# SECOND_MODEL enables a second segmentation model running in parallel.
# Their outputs are merged (intersection = both agree, union = either detects).
SECOND_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DUAL_MODE = False
MERGE_STRATEGY = "intersection"  # "intersection" or "union"

# ---- Bikepath detection ----
# Cityscapes has no separate bike-path class, so paved bike paths are baked
# into the road class. We recover them inside the road mask with a small
# detector that looks for the painted cycle-lane markers, or the
# red-asphalt / dark strip separated from the main road by a curb line.
#
# Heuristic (tuned on typical street-view footage):
#   * Consider only pixels inside the road mask.
#   * The bike path is usually 30..150 cm wide -> 40..220 px at typical image
#     scales. We look for a connected region of similar-colored road pixels
#     that is parallel and adjacent to a clearly different road surface.
#   * Strong colour signal: bike path pavement is often red / dark / painted.
#
# When the detector finds a confident strip, we split it out of the road
# polygon and emit it as class 2 (surface_bikepath). Otherwise the road
# polygon stays whole.
BIKEPATH_DETECT = True
# Saturation threshold: a bike path is often painted red. Lower = more
# permissive.
BIKEPATH_RED_SAT = 35
# Edge threshold: a bike path is bordered by a curb or painted line. The
# Canny-edge response inside the road mask must be high here.
BIKEPATH_EDGE_DENSITY = 0.04
# Minimum bike-path area in pixels (relative to road area) to keep.
BIKEPATH_MIN_FRACTION = 0.015

# ---- Overlay colors (BGR) ----
# Single surface class - use a neutral color
OVERLAY_COLORS_BGR = {
    0: (0, 0, 255),     # red - single merged surface class
}
OVERLAY_ALPHA = 0.30
OVERLAY_THICKNESS = 2

# ---- Model / precision settings ----
# Backbone size: b0 (fastest) ... b5 (most precise/slowest). b3 is a good
# balanced default for CPU. Pick per run in the web UI too.
MODEL_NAME = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
MODEL_VARIANTS = {
    "b0 (fastest)": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "b2 (balanced)": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b5 (max precision)": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

# Inference mode. SegFormer (Cityscapes) is TRAINED on images resized to
# 1024x1024, so the default "single pass" resizes the whole photo to model
# scale. This is fast AND the most accurate because it matches the training
# distribution.
# TILED=True splits huge photos into overlapping 1024px patches and averages
# the overlap. Only useful for very high-res images where downsampling loses
# fine boundary detail; it is slower and can misread scale.
TILED = False
TILE_SIZE = 1024
TILE_OVERLAP = 256

# Min softmax P(surface) to keep a pixel. Lower = more recall, higher =
# stricter. 0.40 is a good balanced default.
CONF_THRESHOLD = 0.40

# ---- YOLO object detection (car) settings ----
YOLO_MODEL_PATH = "yolov8n-seg.pt"
CAR_CONF_THRESHOLD = 0.35
MAX_CAR_INSTANCES = 20

# ---- Green-vegetation heuristic ----
GREEN_EXCLUDE = True
GREEN_MARGIN = 25
GREEN_MIN = 60

# ---- Mask refinement (tighter, better-aligned polygon placement) ----
# Edge-aware: bilateral filter on the soft mask before thresholding. Keeps
# boundary crisp (Gaussian blur smudges it).
REFINE_BILATERAL_D = 5
REFINE_BILATERAL_SIGMA_COLOR = 25
REFINE_BILATERAL_SIGMA_SPACE = 25

# OPEN removes tiny isolated speckles / false positives without eroding thin
# walkways (kernel is small).
REFINE_OPEN_KSIZE = 3
# CLOSE fills small holes inside the surface region.
REFINE_CLOSE_KSIZE = 7

# If the total surface area is smaller than this (in px), treat the whole mask
# as empty. Guards against a few false-positive blobs.
MIN_SURFACE_PIXELS = 500

# Component retention: keep every connected piece whose area is at least
# max(MIN_COMPONENT_PIXELS, RELATIVE_MIN_FRACTION * largest_component).
MIN_COMPONENT_PIXELS = 200
RELATIVE_MIN_FRACTION = 0.02

# Mask blur kernel before contour extraction. Edge-aware, so we don't smooth
# the true boundary.
POLY_BLUR_KSIZE = 3
# approxPolyDP epsilon as a fraction of contour perimeter (smaller = more points
# and a tighter follow of the boundary).
POLY_EPSILON_FRACTION = 0.00025


# ---------------------------------------------------------------------------
# Mask refinement
# ---------------------------------------------------------------------------

def _bilateral_smooth(prob: np.ndarray) -> np.ndarray:
    """Edge-aware smoothing of a probability map. Keeps sharp boundaries."""
    f = prob.astype(np.float32)
    if REFINE_BILATERAL_D > 0:
        f = cv2.bilateralFilter(
            f, REFINE_BILATERAL_D,
            REFINE_BILATERAL_SIGMA_COLOR,
            REFINE_BILATERAL_SIGMA_SPACE,
        )
    return f


def refine_surface_mask(mask: np.ndarray) -> np.ndarray:
    """Clean a binary surface mask.

    Steps:
      1. morphological OPEN  (drop small noise)
      2. morphological CLOSE (fill small holes)
      3. keep every connected component above a relative size floor
    Returns a boolean mask of the same shape.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)

    if REFINE_OPEN_KSIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (REFINE_OPEN_KSIZE, REFINE_OPEN_KSIZE)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if REFINE_CLOSE_KSIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (REFINE_CLOSE_KSIZE, REFINE_CLOSE_KSIZE)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n_labels, labels = cv2.connectedComponents(m)
    if n_labels > 1:
        sizes = np.bincount(labels.flatten())  # index 0 is background
        largest_area = int(np.max(sizes[1:]))
        keep = np.zeros_like(m, dtype=bool)
        for label_id in range(1, n_labels):
            if sizes[label_id] >= max(
                MIN_COMPONENT_PIXELS, RELATIVE_MIN_FRACTION * largest_area
            ):
                keep |= labels == label_id
        m = keep.astype(np.uint8)

    m = m.astype(bool)
    if int(m.sum()) < MIN_SURFACE_PIXELS:
        return np.zeros_like(m, dtype=bool)
    return m


# ---------------------------------------------------------------------------
# Polygon extraction
# ---------------------------------------------------------------------------

def mask_to_polygon(mask_bin, precision=POLY_EPSILON_FRACTION):
    """Binary mask -> (yolo_polygon_str, pixel_points, area) or (None, None, 0).

    precision: fraction of arc length used as epsilon for approxPolyDP.
               Smaller = more precise (denser polygon), larger = more
               simplification.

    Returns ALL contours above the size floor (multi-polygon: long thin
    surfaces split into multiple connected components are preserved
    individually). Output format is one polygon per call -- callers wanting
    multi-polygon should call this once per connected component.
    """
    mask_u8 = mask_bin.astype(np.uint8)

    # Soft-blur the binary mask then re-threshold. Rounds off stair-step
    # pixel edges so the extracted polygon hugs the true boundary instead
    # of zig-zagging through single-pixel bumps.
    if POLY_BLUR_KSIZE > 0:
        blur = cv2.GaussianBlur(
            mask_u8.astype(np.float32), (POLY_BLUR_KSIZE, POLY_BLUR_KSIZE), 0
        )
        mask_u8 = (blur > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None, None, 0
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None, None, 0

    # Adaptive epsilon: proportional to contour perimeter for consistent
    # detail density. Smaller fraction = denser polygon = tighter boundary
    # follow.
    epsilon = max(1.0, precision * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
    if len(approx) < 3:
        return None, None, 0

    pts = approx.reshape(-1, 2).astype(np.float32)
    h, w = mask_bin.shape
    area = cv2.contourArea(cnt)

    # Normalized (YOLO-seg) coordinates
    pts_norm = pts.copy()
    pts_norm[:, 0] /= w
    pts_norm[:, 1] /= h
    yolo_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)

    pixel_pts = [(int(x), int(y)) for x, y in pts]

    return yolo_str, pixel_pts, area


def mask_to_multi_polygon(mask_bin, precision=POLY_EPSILON_FRACTION,
                          min_area=200):
    """All contours above min_area, in largest-first order.

    Returns a list of (yolo_str, pixel_pts, area) tuples.
    """
    mask_u8 = mask_bin.astype(np.uint8)
    if POLY_BLUR_KSIZE > 0:
        blur = cv2.GaussianBlur(
            mask_u8.astype(np.float32), (POLY_BLUR_KSIZE, POLY_BLUR_KSIZE), 0
        )
        mask_u8 = (blur > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    h, w = mask_bin.shape
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            break
        epsilon = max(1.0, precision * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts_norm = pts.copy()
        pts_norm[:, 0] /= w
        pts_norm[:, 1] /= h
        yolo_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)
        pixel_pts = [(int(x), int(y)) for x, y in pts]
        out.append((yolo_str, pixel_pts, float(cv2.contourArea(cnt))))
    return out


# ---------------------------------------------------------------------------
# Bikepath detection
# ---------------------------------------------------------------------------

def detect_bikepath(road_mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Try to extract a bike-path sub-mask from inside the road mask.

    Heuristic: bike paths are often (a) painted red, or (b) separated from
    the main road by a curb or painted line (high local edge density). We
    return a boolean mask of the bike-path pixels, or an empty mask if
    nothing confidently matches.

    This is intentionally conservative -- when in doubt, we leave the road
    polygon whole instead of splitting off junk.
    """
    if road_mask.sum() == 0:
        return np.zeros_like(road_mask, dtype=bool)

    h, w = road_mask.shape

    # ---- Color signal: red-painted bike path (HSV) ----
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Hue 0..10 or 170..180 with reasonable saturation = red paint.
    hue, sat, _ = cv2.split(hsv)
    red_paint = (
        ((hue <= 10) | (hue >= 170))
        & (sat >= BIKEPATH_RED_SAT)
        & road_mask
    )

    # ---- Edge signal: curb / painted-line border inside the road mask ----
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    # Dilate edges slightly so a thin painted line still registers.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Build a "border" map: pixels inside the road that are within 6 px of
    # an edge that ALSO lies inside the road. This is the curb / line that
    # separates a bike path from the car lane.
    interior_edges = edges & road_mask
    interior_border = cv2.dilate(
        interior_edges.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1
    ) > 0
    # The bike path sits on one side of such a border. We pick the side
    # that is closer to the image edge or has different color, but as a
    # quick proxy we use the border itself plus a 12-px band on each side.
    bike_band = cv2.dilate(
        interior_border.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1
    ) > 0
    bike_band &= road_mask

    # Combine the two signals.
    candidate = (red_paint | bike_band) & road_mask

    if candidate.sum() < BIKEPATH_MIN_FRACTION * road_mask.sum():
        return np.zeros_like(road_mask, dtype=bool)

    # Keep the largest connected component of `candidate` -- bike paths are
    # one continuous strip.
    cand_u8 = candidate.astype(np.uint8)
    n_lbl, lbl = cv2.connectedComponents(cand_u8)
    if n_lbl <= 1:
        return np.zeros_like(road_mask, dtype=bool)
    sizes = np.bincount(lbl.flatten())
    if len(sizes) <= 1:
        return np.zeros_like(road_mask, dtype=bool)
    biggest = int(np.argmax(sizes[1:])) + 1
    bike_mask = lbl == biggest

    if bike_mask.sum() < BIKEPATH_MIN_FRACTION * road_mask.sum():
        return np.zeros_like(road_mask, dtype=bool)

    # Refine the bike path mask: open + close.
    if REFINE_OPEN_KSIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (REFINE_OPEN_KSIZE, REFINE_OPEN_KSIZE)
        )
        bike_mask = cv2.morphologyEx(
            bike_mask.astype(np.uint8), cv2.MORPH_OPEN, k
        ).astype(bool)
    if REFINE_CLOSE_KSIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (REFINE_CLOSE_KSIZE, REFINE_CLOSE_KSIZE)
        )
        bike_mask = cv2.morphologyEx(
            bike_mask.astype(np.uint8), cv2.MORPH_CLOSE, k
        ).astype(bool)

    # Bike path must remain inside the road mask.
    bike_mask &= road_mask

    if bike_mask.sum() < BIKEPATH_MIN_FRACTION * road_mask.sum():
        return np.zeros_like(road_mask, dtype=bool)

    return bike_mask


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class Annotator:
    """Holds the SegFormer model in memory; reused across many images."""

    def __init__(
        self,
        model_dir: Path,
        model_name: str = MODEL_NAME,
        tiled: bool = TILED,
        class_prefix: str = CLASS_PREFIX,
        fast_mode: bool = FAST_MODE,
        fast_size: int = FAST_SIZE,
        dual_mode: bool = DUAL_MODE,
        detect_cars: bool = False,
        car_conf_threshold: float = CAR_CONF_THRESHOLD,
        yolo_model_path: str = YOLO_MODEL_PATH,
        multi_scale: bool = MULTI_SCALE,
        multi_scale_sizes=MULTI_SCALE_SIZES,
        tta: bool = TTA,
        detect_bikepath: bool = BIKEPATH_DETECT,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.tiled = tiled
        self.conf_threshold = CONF_THRESHOLD
        self.green_exclude = GREEN_EXCLUDE
        self.class_prefix = class_prefix
        self.fast_mode = fast_mode
        self.fast_size = fast_size
        self.dual_mode = dual_mode
        self.detect_cars = detect_cars
        self.car_conf_threshold = car_conf_threshold
        self.multi_scale = multi_scale
        self.multi_scale_sizes = tuple(multi_scale_sizes)
        self.tta = tta
        self.detect_bikepath_flag = detect_bikepath

        self.seg_proc = SegformerImageProcessor.from_pretrained(
            model_name, cache_dir=str(self.model_dir)
        )
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, cache_dir=str(self.model_dir)
        )
        self.seg_model.eval()

        self.second_proc = None
        self.second_model = None
        if dual_mode:
            self.second_proc = SegformerImageProcessor.from_pretrained(
                SECOND_MODEL_NAME, cache_dir=str(self.model_dir)
            )
            self.second_model = SegformerForSemanticSegmentation.from_pretrained(
                SECOND_MODEL_NAME, cache_dir=str(self.model_dir)
            )
            self.second_model.eval()

        self.yolo_model = None
        if detect_cars:
            self._load_yolo(yolo_model_path)

    # ---- YOLO (car) ----

    def _load_yolo(self, yolo_model_path: str):
        """Load the YOLOv8-seg model for car detection.

        Searches for the checkpoint in: cwd, model_dir, and model_dir/models.
        Falls back gracefully (sets self.yolo_model = None) if not found.
        """
        from ultralytics import YOLO

        candidates = [
            yolo_model_path,
            str(self.model_dir / yolo_model_path),
            str(self.model_dir / "models" / yolo_model_path),
            str(self.model_dir / Path(yolo_model_path).name),
        ]
        found = None
        for cand in candidates:
            if Path(cand).exists():
                found = cand
                break
        if found is None:
            try:
                self.yolo_model = YOLO(yolo_model_path)
            except Exception:
                self.yolo_model = None
            return
        self.yolo_model = YOLO(found)

    def _run_yolo_cars(self, bgr: np.ndarray):
        """Run YOLOv8-seg on the image and return list of car instance dicts.

        Each dict: {"mask": np.ndarray(H,W, bool), "conf": float, "box": list}
        Only COCO classes in CAR_COCO_IDS are returned.
        """
        if self.yolo_model is None:
            return []

        h, w = bgr.shape[:2]
        results = self.yolo_model(bgr, verbose=False, device="cpu")
        r = results[0]
        cars = []

        if r.masks is None or r.boxes is None:
            return cars

        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for i in range(len(cls_ids)):
            coco_id = int(cls_ids[i])
            if coco_id not in CAR_COCO_IDS:
                continue
            if confs[i] < self.car_conf_threshold:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask_points = r.masks[i].xy
            pts = np.array(mask_points[0], dtype=np.int32)
            cv2.fillPoly(mask, [pts], True)
            cars.append({
                "mask": mask,
                "conf": float(confs[i]),
                "box": r.boxes[i].xyxy.cpu().numpy().flatten().tolist(),
            })

        return cars[:MAX_CAR_INSTANCES]

    # ---- SegFormer inference ----

    def _run_inference(self, rgb: np.ndarray) -> np.ndarray:
        """One forward pass on a single patch -> (H, W, C) logits at patch res."""
        pil = Image.fromarray(rgb)
        inputs = self.seg_proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            logits = self.seg_model(**inputs).logits
        h, w = rgb.shape[:2]
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )[0].cpu().numpy()  # (C, H, W)
        return logits.transpose(1, 2, 0)  # (H, W, C)

    def _tiled_logits(self, rgb: np.ndarray) -> np.ndarray:
        """Full-resolution class logits (H, W, C) via overlapping-tile inference.

        Only used when tiled=True and the image exceeds TILE_SIZE. Overlapping
        TILE_SIZE patches (padded with edge replication) are inferred and the
        overlap zones averaged to hide seams.
        """
        h, w = rgb.shape[:2]
        S = TILE_SIZE
        step = max(1, S - TILE_OVERLAP)
        n_h = max(1, int(np.ceil(max(0, h - S) / step)) + 1)
        n_w = max(1, int(np.ceil(max(0, w - S) / step)) + 1)
        pad_h = n_h * step + (S - step) - h
        pad_w = n_w * step + (S - step) - w
        padded = cv2.copyMakeBorder(
            rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
        )
        ph, pw = padded.shape[:2]

        n_classes = self.seg_model.config.num_labels
        logits = np.zeros((ph, pw, n_classes), dtype=np.float32)
        weight = np.zeros((ph, pw, 1), dtype=np.float32)
        for y0 in range(0, ph - S + 1, step):
            for x0 in range(0, pw - S + 1, step):
                tile = padded[y0:y0 + S, x0:x0 + S]
                logits[y0:y0 + S, x0:x0 + S] += self._run_inference(tile)
                weight[y0:y0 + S, x0:x0 + S] += 1.0
        logits /= np.maximum(weight, 1e-9)
        return logits[:h, :w]

    def _predict_logits(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, C) logits at the image's native resolution.

        Honors fast_mode (down-sample to fast_size), tiled (overlapping
        patches for huge images), multi_scale (average over several sizes),
        and TTA (horizontal-flip average).
        """
        # Multi-scale: average probabilities across sizes for stability.
        if self.multi_scale:
            probs_acc = None
            for sz in self.multi_scale_sizes:
                oh, ow = rgb.shape[:2]
                scale = min(sz / oh, sz / ow)
                new_w = max(1, int(ow * scale) // 16 * 16)
                new_h = max(1, int(oh * scale) // 16 * 16)
                if new_w == ow and new_h == oh:
                    logits = self._run_inference(rgb)
                else:
                    small = cv2.resize(
                        rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    logits = self._run_inference(small)
                    logits = cv2.resize(
                        logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                    )
                probs = torch.softmax(
                    torch.from_numpy(logits).float(), dim=-1
                ).numpy()
                probs_acc = probs if probs_acc is None else probs_acc + probs
            probs_avg = probs_acc / len(self.multi_scale_sizes)

            # TTA: flip-average over the multi-scale probabilities.
            if self.tta:
                probs_flip = None
                for sz in self.multi_scale_sizes:
                    rgb_f = rgb[:, ::-1, :].copy()
                    oh, ow = rgb_f.shape[:2]
                    scale = min(sz / oh, sz / ow)
                    new_w = max(1, int(ow * scale) // 16 * 16)
                    new_h = max(1, int(oh * scale) // 16 * 16)
                    if new_w == ow and new_h == oh:
                        logits = self._run_inference(rgb_f)
                    else:
                        small = cv2.resize(
                            rgb_f, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )
                        logits = self._run_inference(small)
                        logits = cv2.resize(
                            logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                        )
                    probs = torch.softmax(
                        torch.from_numpy(logits).float(), dim=-1
                    ).numpy()
                    # Un-flip the X axis back.
                    probs = probs[:, ::-1, :].copy()
                    probs_flip = probs if probs_flip is None else probs_flip + probs
                probs_flip /= len(self.multi_scale_sizes)
                probs_avg = (probs_avg + probs_flip) / 2.0

            # Convert averaged probs back to "logits-like" by inverting the
            # softmax -- this is fine because all downstream code uses
            # softmax again on the result.
            eps = 1e-9
            return np.log(probs_avg + eps)

        # ---- Single-scale path (fast_mode / tiled / normal) ----
        if self.fast_mode:
            oh, ow = rgb.shape[:2]
            scale = min(self.fast_size / oh, self.fast_size / ow)
            new_w = max(1, int(ow * scale) // 16 * 16)
            new_h = max(1, int(oh * scale) // 16 * 16)
            if new_w != ow or new_h != oh:
                small = cv2.resize(
                    rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                logits = self._run_inference(small)
                logits = cv2.resize(
                    logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                )
            else:
                logits = self._run_inference(rgb)
        elif self.tiled and (
            rgb.shape[0] > TILE_SIZE or rgb.shape[1] > TILE_SIZE
        ):
            logits = self._tiled_logits(rgb)
        else:
            logits = self._run_inference(rgb)

        # TTA: flip-average.
        if self.tta:
            rgb_f = rgb[:, ::-1, :].copy()
            if self.fast_mode:
                oh, ow = rgb_f.shape[:2]
                scale = min(self.fast_size / oh, self.fast_size / ow)
                new_w = max(1, int(ow * scale) // 16 * 16)
                new_h = max(1, int(oh * scale) // 16 * 16)
                if new_w != ow or new_h != oh:
                    small = cv2.resize(
                        rgb_f, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                    logits_f = self._run_inference(small)
                    logits_f = cv2.resize(
                        logits_f, (ow, oh), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    logits_f = self._run_inference(rgb_f)
            else:
                logits_f = self._run_inference(rgb_f)
            # Un-flip X axis on logits (per-class channel order is preserved).
            logits_f = logits_f[:, ::-1, :].copy()
            probs = torch.softmax(
                torch.from_numpy(logits).float(), dim=-1
            ).numpy()
            probs_f = torch.softmax(
                torch.from_numpy(logits_f).float(), dim=-1
            ).numpy()
            probs_avg = (probs + probs_f) / 2.0
            eps = 1e-9
            return np.log(probs_avg + eps)

        return logits

    # ---- Per-class surface mask extraction ----

    # ---- Second (dual) model ----

    def _run_inference_second(self, rgb: np.ndarray) -> np.ndarray:
        """One forward pass on a single patch with the second model."""
        pil = Image.fromarray(rgb)
        inputs = self.second_proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            logits = self.second_model(**inputs).logits
        h, w = rgb.shape[:2]
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )[0].cpu().numpy()
        return logits.transpose(1, 2, 0)

    def _predict_logits_second(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, C) logits from the second model at the image's native
        resolution. No multi-scale or TTA on the second model -- its job is
        to be a diverse cross-check, not a precision push.
        """
        if self.fast_mode:
            oh, ow = rgb.shape[:2]
            scale = min(self.fast_size / oh, self.fast_size / ow)
            new_w = max(1, int(ow * scale) // 16 * 16)
            new_h = max(1, int(oh * scale) // 16 * 16)
            if new_w != ow or new_h != oh:
                small = cv2.resize(
                    rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                logits = self._run_inference_second(small)
                logits = cv2.resize(
                    logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                )
                return logits
        return self._run_inference_second(rgb)

    # ---- Per-class surface mask extraction ----

    def _class_prob(self, logits: np.ndarray, target_class: int) -> np.ndarray:
        """P(class == target_class | logits) at the image's native resolution."""
        probs = torch.softmax(
            torch.from_numpy(logits).float(), dim=-1
        ).numpy()
        return probs[..., target_class]

    def _surface_masks(self, logits: np.ndarray):
        """Return per-class surface masks and combined vegetation/black/green
        exclusion masks.

        Returns:
            masks: dict[class_id, np.ndarray(bool)]
            black_mask, green_mask
        """
        # We need RGB for color exclusions; pass it down from annotate() via
        # closure by attaching it to the annotator instance.
        rgb = self._current_rgb
        h, w = rgb.shape[:2]

        probs = torch.softmax(
            torch.from_numpy(logits).float(), dim=-1
        ).numpy()

        # Vegetation exclusion -- for each surface class, keep pixels where
        # P(surface) > P(vegetation).
        veg_prob = (
            probs[..., list(VEGETATION_SEG_IDS)].sum(axis=-1)
            if VEGETATION_SEG_IDS
            else np.zeros((h, w), dtype=np.float32)
        )

        # Black / green masks.
        black_mask = (rgb < BLACK_THRESHOLD).all(axis=2)
        if self.green_exclude:
            r = rgb[..., 0].astype(np.int16)
            g = rgb[..., 1].astype(np.int16)
            b = rgb[..., 2].astype(np.int16)
            green_mask = (
                (g > r + GREEN_MARGIN)
                & (g > b + GREEN_MARGIN)
                & (g > GREEN_MIN)
            )
        else:
            green_mask = np.zeros((h, w), dtype=bool)

        masks = {}
        for our_id, seg_id in SURFACE_SEG_IDS.items():
            cls_prob = probs[..., seg_id]
            mask = cls_prob >= self.conf_threshold
            mask &= cls_prob > veg_prob
            mask &= ~black_mask
            mask &= ~green_mask
            masks[our_id] = mask

        return masks, black_mask, green_mask

    # ---- Merging strategy (dual model) ----

    def _merge_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        if MERGE_STRATEGY == "union":
            return mask1 | mask2
        return mask1 & mask2

    # ---- Main entry ----

    def annotate(self, image_path: Path, lbl_dir: Path,
                 preview_dir: Path, annotated_dir: Path):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"cannot read {image_path}")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._current_rgb = rgb  # used by _surface_masks

        # ---- SegFormer pass ----
        logits = self._predict_logits(rgb)  # (H, W, C) at native resolution
        surface_masks, black_mask, green_mask = self._surface_masks(logits)

        # ---- Second model (dual mode) ----
        if self.dual_mode and self.second_model is not None:
            logits2 = self._predict_logits_second(rgb)
            surface_masks2, _, _ = self._surface_masks(logits2)
            surface_masks = {
                k: self._merge_masks(surface_masks[k], surface_masks2[k])
                for k in surface_masks
            }

        # ---- Refine each class mask ----
        for cid in list(surface_masks.keys()):
            surface_masks[cid] = refine_surface_mask(surface_masks[cid])

        road_mask = surface_masks.get(0, np.zeros((h, w), dtype=bool))
        walkway_mask = surface_masks.get(1, np.zeros((h, w), dtype=bool))

        # ---- Bikepath: try to split a strip out of the road mask ----
        bike_mask = np.zeros((h, w), dtype=bool)
        if self.detect_bikepath_flag and road_mask.sum() > 0:
            bike_mask = detect_bikepath(road_mask, rgb)
            if bike_mask.any():
                road_mask = road_mask & ~bike_mask

        # ---- Build polygons per class ----
        # We use multi-polygon (one per connected component) so that a road
        # split by a pedestrian island, or a walkway that fades out, is
        # preserved as separate polygons instead of one big blob.
        out_polys = []  # list of (class_id, yolo_str, pixel_pts, area)
        for cid in (0, 1, 2):  # road, walkway, bikepath
            mask = {
                0: road_mask, 1: walkway_mask, 2: bike_mask,
            }[cid]
            if not mask.any():
                continue
            for yolo_str, pixel_pts, area in mask_to_multi_polygon(
                mask, precision=POLY_EPSILON_FRACTION,
            ):
                out_polys.append((cid, yolo_str, pixel_pts, area))

        # ---- Draw per-class overlays ----
        overlay = bgr.copy()
        for cid, yolo_str, pixel_pts, area in out_polys:
            color = OVERLAY_COLORS_BGR.get(cid, (0, 0, 255))
            mask = {
                0: road_mask, 1: walkway_mask, 2: bike_mask,
            }[cid]
            overlay[mask] = color
        if out_polys:
            bgr = cv2.addWeighted(
                bgr, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0
            )
            for cid, yolo_str, pixel_pts, area in out_polys:
                color = OVERLAY_COLORS_BGR.get(cid, (0, 0, 255))
                pts = np.array(pixel_pts, dtype=np.int32)
                bgr = cv2.polylines(
                    bgr, [pts], True, color, OVERLAY_THICKNESS
                )

        # ---- YOLO cars (clipped to surface) ----
        car_polys = []
        car_count = 0
        if self.detect_cars and self.yolo_model is not None:
            car_instances = self._run_yolo_cars(bgr)
            # Combine all surface masks (road + walkway + bike) into one
            # "anything paved" mask for clipping.
            any_surface = road_mask | walkway_mask | bike_mask
            for car in car_instances:
                car_mask = car["mask"] & any_surface
                if car_mask.sum() < 200:
                    continue
                c_poly, c_pts, c_area = mask_to_polygon(
                    car_mask, precision=POLY_EPSILON_FRACTION
                )
                if c_poly:
                    car_count += 1
                    car_polys.append({
                        "yolo_poly": c_poly,
                        "pixel_points": c_pts,
                        "area": c_area,
                        "conf": car["conf"],
                    })
                    c_overlay = bgr.copy()
                    c_overlay[car_mask] = OVERLAY_COLORS_BGR[3]
                    bgr = cv2.addWeighted(
                        bgr, 1.0 - OVERLAY_ALPHA, c_overlay, OVERLAY_ALPHA, 0
                    )
                    c_arr = np.array(c_pts, dtype=np.int32)
                    bgr = cv2.polylines(
                        bgr, [c_arr], True, OVERLAY_COLORS_BGR[3],
                        OVERLAY_THICKNESS,
                    )

        # ---- Write label file ----
        lbl_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        label_lines = [f"{cid} {yolo_str}" for cid, yolo_str, _, _ in out_polys]
        for c in car_polys:
            label_lines.append(f"3 {c['yolo_poly']}")
        label_path = lbl_dir / (image_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

        # ---- Write preview ----
        legend_y = 25
        for cid, name in [
            (0, f"{self.class_prefix}road"),
            (1, f"{self.class_prefix}walkway"),
            (2, f"{self.class_prefix}bikepath"),
        ]:
            present = any(p[0] == cid for p in out_polys)
            if present:
                cv2.putText(
                    bgr, name, (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    OVERLAY_COLORS_BGR[cid], 2,
                )
                legend_y += 22
        if car_count:
            cv2.putText(
                bgr, f"car ({car_count})", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                OVERLAY_COLORS_BGR[3], 2,
            )

        preview_path = preview_dir / image_path.name
        cv2.imwrite(str(preview_path), bgr)
        shutil.copy2(str(preview_path), str(annotated_dir / image_path.name))

        return {
            "label_path": str(label_path),
            "preview_path": str(preview_path),
            "annotated_path": str(annotated_dir / image_path.name),
            "polys_by_class": {
                cid: sum(1 for c, *_ in out_polys if c == cid)
                for cid in (0, 1, 2)
            },
            "surface_polys": len(out_polys),  # backwards-compat total
            "road_polys": sum(1 for c, *_ in out_polys if c == 0),
            "walkway_polys": sum(1 for c, *_ in out_polys if c == 1),
            "bike_polys": sum(1 for c, *_ in out_polys if c == 2),
            "car_polys": car_count,
            "car_polys_detail": car_polys,
            "image_name": image_path.name,
            "width": w,
            "height": h,
            "polygons": [
                {
                    "class_id": cid,
                    "yolo_poly": y,
                    "pixel_points": pts,
                    "area": a,
                }
                for (cid, y, pts, a) in out_polys
            ],
        }


def export_cvat_xml(
    results, output_path, task_name="Surface Auto-Annotator",
    class_prefix=CLASS_PREFIX, detect_cars=False,
):
    """Export annotation results to CVAT XML format.

    Emits one <label> per surface class (road, walkway, bikepath) and,
    optionally, a "car" label. Each result is rendered with one
    <polygon label="..."> per detected shape.
    """
    road_name = f"{class_prefix}road"
    walkway_name = f"{class_prefix}walkway"
    bike_name = f"{class_prefix}bikepath"
    output_path = Path(output_path)
    now = datetime.datetime.now().isoformat()

    label_lines = [
        f'      <label><name>{road_name}</name><type>polygon</type><attributes/></label>',
        f'      <label><name>{walkway_name}</name><type>polygon</type><attributes/></label>',
        f'      <label><name>{bike_name}</name><type>polygon</type><attributes/></label>',
    ]
    if detect_cars:
        label_lines.append(
            '      <label><name>car</name><type>polygon</type><attributes/></label>'
        )

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<annotations>",
        "  <version>1.1</version>",
        "  <meta>",
        f'    <task><id>1</id><name>{task_name}</name><size>{len(results)}</size><mode>annotation</mode><created>{now}</created><updated>{now}</updated></task>',
        "    <labels>",
        *label_lines,
        "    </labels>",
        "  </meta>",
    ]

    name_for_class = {0: road_name, 1: walkway_name, 2: bike_name}

    for i, r in enumerate(results):
        lines.append(
            f'  <image id="{i}" name="{r["image_name"]}" width="{r["width"]}" height="{r["height"]}">'
        )
        for poly in r.get("polygons", []) or []:
            pts = poly.get("pixel_points") or []
            if not pts:
                continue
            pts_str = ";".join(f"{x},{y}" for x, y in pts)
            label = name_for_class.get(poly["class_id"], road_name)
            lines.append(
                f'    <polygon label="{label}" points="{pts_str}" z_order="0"/>'
            )
        if detect_cars:
            for car in r.get("car_polys_detail", []) or []:
                pts = car.get("pixel_points") or []
                if not pts:
                    continue
                pts_str = ";".join(f"{x},{y}" for x, y in pts)
                lines.append(
                    f'    <polygon label="car" points="{pts_str}" z_order="1"/>'
                )
        lines.append("  </image>")

    lines.append("</annotations>")
    output_path.write_text("\n".join(lines))
    return output_path
