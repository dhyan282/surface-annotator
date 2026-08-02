"""
Shared annotator core - surface + car.
Detects paved surface classes (road, walkway/sidewalk, bikepath) via SegFormer /
Cityscapes, merged into a single "surface_surface" polygon. Optionally detects
cars via YOLOv8-seg and emits them as separate "car" parts that overlap the
surface region.

Output classes:
    0  surface_surface  (road + walkway + bikepath merged into ONE polygon)
    1  car              (YOLOv8-seg instance segmentation, overlapping surface)

Cityscapes class coverage -> one output class:
    0  road          (also covers bike paths, which Cityscapes does not model
                      separately -> paved surfaces)
    1  sidewalk      (walkway)
Both merge into our single class 0 "surface_surface".
"""

from pathlib import Path
import shutil
import datetime
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ---- Surface class + car class ----
CLASS_PREFIX = "surface_"

# COCO class ids that we map to our "car" class (id 1).
#  2 = car, 5 = bus, 7 = truck, 3 = motorbike
# Cars, buses, trucks (and optionally motorbike) become "car" parts.
CAR_COCO_IDS = {2, 5, 7}

# Class definitions written to classes.txt / dataset.yaml
CLASSES = [
    (0, f"{CLASS_PREFIX}surface"),  # road + walkway + bikepath
    (1, "car"),
]

# SegFormer (Cityscapes) class ids mapped to our single surface class.
#   0 = road       (Cityscapes has no separate bikepath -> paved/bike paths count as road)
#   1 = sidewalk   (walkway)
# Only paved surfaces are included; vegetation/terrain are excluded.
SURFACE_SEG_IDS = {0: 0, 1: 0}  # segformer_id -> our_id

BLACK_THRESHOLD = 15  # RGB values below this in ALL channels = "black" (excluded)

# ---- Speed settings ----
# FAST_MODE=True resizes images to FAST_SIZE before inference for speed.
# This trades some precision for much faster annotation.
FAST_MODE = True
FAST_SIZE = 640

# ---- Dual-model settings ----
# SECOND_MODEL enables a second segmentation model running in parallel.
# Their outputs are merged (intersection = both agree, union = either detects).
# Using a different model size provides diversity through different capacity.
SECOND_MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DUAL_MODE = False
MERGE_STRATEGY = "intersection"  # "intersection" or "union"

# ---- Overlay colors (BGR) ----
OVERLAY_COLOR = (0, 0, 255)         # red fill for surface
OVERLAY_ALPHA = 0.3
OVERLAY_OUTLINE_COLOR = (0, 0, 255)  # red outline for surface
OVERLAY_THICKNESS = 2

# Car part overlay (blue)
CAR_OVERLAY_COLOR = (255, 0, 0)          # blue fill for car
CAR_OVERLAY_OUTLINE_COLOR = (255, 0, 0)  # blue outline for car
CAR_OVERLAY_THICKNESS = 2
CAR_OVERLAY_ALPHA = 0.4

# ---- Model / precision settings ----
# Backbone size: b0 (fastest) ... b5 (most precise/slowest). b2 is a good
# balanced default for CPU. Pick per run in the web UI too.
MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
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

# Min softmax P(surface) to keep a pixel. Lower = more recall (fixes missed
# walkways), higher = stricter. 0.4 is a good balanced default.
CONF_THRESHOLD = 0.4

# ---- YOLO object detection (car) settings ----
# YOLO_MODEL_PATH: relative filename of the YOLOv8-seg checkpoint.
# The model is expected in the model_dir passed to Annotator.
# If the file does not exist, car detection is silently skipped.
YOLO_MODEL_PATH = "yolov8n-seg.pt"
# Confidence threshold for car detection (YOLO instance predictions).
CAR_CONF_THRESHOLD = 0.35
# Maximum number of car instances to extract per image.
MAX_CAR_INSTANCES = 20

# Cityscapes "vegetation" class. Pixels where P(vegetation) >= P(surface) are
# excluded, so grass/shrubs bleeding into the road don't get annotated.
VEGETATION_SEG_IDS = {8}

# Heuristic: pixels that are strongly green are almost never paved surface.
# Excludes grass/leaf false positives even when the model is uncertain.
GREEN_EXCLUDE = True
GREEN_MARGIN = 25
GREEN_MIN = 60

BLACK_THRESHOLD = 15  # RGB values below this in ALL channels = "black" (excluded)

# ---- Mask refinement (tighter, better-aligned polygon placement) ----
# Open:   remove tiny isolated speckles / false positives.
# Close:  fill small holes inside the surface region.
# Both use an elliptical structuring element to keep edges natural.
# OPEN is small (3) so thin walkways are not eroded away.
REFINE_OPEN_KSIZE = 3
REFINE_CLOSE_KSIZE = 7

# If the total surface area is smaller than this (in px), treat the whole mask
# as empty. Guards against a few false-positive blobs.
MIN_SURFACE_PIXELS = 500

# Component retention: keep every connected piece whose area is at least
# max(MIN_COMPONENT_PIXELS, RELATIVE_MIN_FRACTION * largest_component).
# Unlike "keep only the largest", this preserves long thin roads that split
# into several segments (a common precision loss).
MIN_COMPONENT_PIXELS = 200
RELATIVE_MIN_FRACTION = 0.02

# Gaussian blur radius applied to the mask before contour extraction.
# Smooths stair-step jaggies so the polygon follows the true boundary closely.
POLY_BLUR_KSIZE = 5
# approxPolyDP epsilon as a fraction of contour perimeter (smaller = more points).
POLY_EPSILON_FRACTION = 0.0005


def refine_surface_mask(mask: np.ndarray) -> np.ndarray:
    """Clean a binary surface mask in-place style.

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
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REFINE_OPEN_KSIZE, REFINE_OPEN_KSIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if REFINE_CLOSE_KSIZE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REFINE_CLOSE_KSIZE, REFINE_CLOSE_KSIZE))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # Keep every component that is meaningfully large. Dropping everything but
    # the single largest component deletes long, thin roads that split into
    # multiple segments -- a real precision loss for this use case.
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


def mask_to_polygon(mask_bin, precision=0.001):
    """Binary mask -> (yolo_polygon_str, pixel_points, area) or (None, None, 0).

    precision: fraction of arc length used as epsilon for approxPolyDP.
               Smaller = more precise, larger = more simplification.
    """
    mask_u8 = mask_bin.astype(np.uint8)

    # Gaussian-blur the binary mask then re-threshold. This rounds off the
    # stair-step pixel edges so the extracted polygon hugs the true boundary
    # instead of zig-zagging through single-pixel bumps.
    if POLY_BLUR_KSIZE > 0:
        blur = cv2.GaussianBlur(mask_u8.astype(np.float32), (POLY_BLUR_KSIZE, POLY_BLUR_KSIZE), 0)
        mask_u8 = (blur > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, 0
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return None, None, 0

    # Adaptive epsilon: proportional to contour perimeter for consistent detail
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

    # Pixel coordinates for CVAT export
    pixel_pts = [(int(x), int(y)) for x, y in pts]

    return yolo_str, pixel_pts, area


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
            # Try loading by name (YOLO will look in cache / hub)
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
        padded = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
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
        """(H, W, C) logits at the image's native resolution."""
        if self.fast_mode:
            oh, ow = rgb.shape[:2]
            scale = min(self.fast_size / oh, self.fast_size / ow)
            new_w = max(1, int(ow * scale) // 16 * 16)
            new_h = max(1, int(oh * scale) // 16 * 16)
            if new_w != ow or new_h != oh:
                small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logits = self._run_inference(small)
                logits = cv2.resize(
                    logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                )
                return logits
        if self.tiled and (rgb.shape[0] > TILE_SIZE or rgb.shape[1] > TILE_SIZE):
            return self._tiled_logits(rgb)
        return self._run_inference(rgb)

    def _predict_logits_second(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, C) logits from the second model at the image's native resolution."""
        if self.fast_mode:
            oh, ow = rgb.shape[:2]
            scale = min(self.fast_size / oh, self.fast_size / ow)
            new_w = max(1, int(ow * scale) // 16 * 16)
            new_h = max(1, int(oh * scale) // 16 * 16)
            if new_w != ow or new_h != oh:
                small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logits = self._run_inference_second(small)
                logits = cv2.resize(
                    logits, (ow, oh), interpolation=cv2.INTER_LINEAR
                )
                return logits
        return self._run_inference_second(rgb)

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

    def _merge_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Merge two binary masks using the configured strategy."""
        if MERGE_STRATEGY == "union":
            return mask1 | mask2
        return mask1 & mask2

    def annotate(self, image_path: Path, lbl_dir: Path, preview_dir: Path, annotated_dir: Path):
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"cannot read {image_path}")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --- SegFormer: road + walkway + bikepath -> single surface mask ---
        logits = self._predict_logits(rgb)  # full-res (H, W, C)

        # Convert to probabilities and threshold, instead of hard argmax. This
        # rejects low-confidence boundary pixels for a tighter, cleaner mask.
        probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
        surface_ids = list(SURFACE_SEG_IDS.keys())
        surface_prob = probs[..., surface_ids].sum(axis=-1)
        surface_mask = surface_prob >= self.conf_threshold

        # Exclude pixels the model thinks are vegetation (grass/shrubs).
        if VEGETATION_SEG_IDS:
            veg_prob = probs[..., list(VEGETATION_SEG_IDS)].sum(axis=-1)
            surface_mask = surface_mask & (surface_prob >= veg_prob)

        # Exclude black regions (e.g., car bonnets that may be misclassified)
        black_mask = (rgb < BLACK_THRESHOLD).all(axis=2)
        surface_mask = surface_mask & ~black_mask

        # Heuristic: strongly green pixels are never paved surface.
        if self.green_exclude:
            r = rgb[..., 0].astype(np.int16)
            g = rgb[..., 1].astype(np.int16)
            b = rgb[..., 2].astype(np.int16)
            green_mask = (g > r + GREEN_MARGIN) & (g > b + GREEN_MARGIN) & (g > GREEN_MIN)
            surface_mask = surface_mask & ~green_mask

        # --- Second model (dual mode) ---
        if self.dual_mode and self.second_model is not None:
            logits2 = self._predict_logits_second(rgb)
            probs2 = torch.softmax(torch.from_numpy(logits2).float(), dim=-1).numpy()
            surface_ids2 = list(SURFACE_SEG_IDS.keys())
            surface_prob2 = probs2[..., surface_ids2].sum(axis=-1)
            surface_mask2 = surface_prob2 >= self.conf_threshold
            if VEGETATION_SEG_IDS:
                veg_prob2 = probs2[..., list(VEGETATION_SEG_IDS)].sum(axis=-1)
                surface_mask2 = surface_mask2 & (surface_prob2 >= veg_prob2)
            surface_mask2 = surface_mask2 & ~black_mask
            if self.green_exclude:
                surface_mask2 = surface_mask2 & ~green_mask
            surface_mask = self._merge_masks(surface_mask, surface_mask2)

        # Clean the mask for tighter, better-aligned polygon placement.
        surface_mask = refine_surface_mask(surface_mask)

        # Single polygon from merged mask (higher precision)
        yolo_poly, pixel_pts, area = mask_to_polygon(
            surface_mask, precision=POLY_EPSILON_FRACTION
        )

        label_lines = []
        car_polys = []
        car_count = 0

        if yolo_poly:
            label_lines.append(f"0 {yolo_poly}")
            # Red fill overlay on the surface mask
            overlay = bgr.copy()
            overlay[surface_mask] = OVERLAY_COLOR
            bgr = cv2.addWeighted(bgr, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
            # Red polygon outline on top
            pts = np.array(pixel_pts, dtype=np.int32)
            bgr = cv2.polylines(bgr, [pts], True, OVERLAY_OUTLINE_COLOR, OVERLAY_THICKNESS)

        # --- YOLO: car detection on overlapping surface region ---
        if self.detect_cars and self.yolo_model is not None:
            car_instances = self._run_yolo_cars(bgr)
            for car in car_instances:
                car_mask = car["mask"]
                # Clip car polygon to the surface mask area (only keep parts
                # that are actually on the road / walkway).
                car_mask = car_mask & surface_mask
                if car_mask.sum() < 200:
                    continue
                # Convert car mask -> YOLO-seg polygon
                c_poly, c_pts, c_area = mask_to_polygon(car_mask, precision=0.001)
                if c_poly:
                    car_count += 1
                    label_lines.append(f"1 {c_poly}")
                    car_polys.append({
                        "yolo_poly": c_poly,
                        "pixel_points": c_pts,
                        "area": c_area,
                        "conf": car["conf"],
                    })
                    # Blue overlay for cars
                    c_overlay = bgr.copy()
                    c_overlay[car_mask] = CAR_OVERLAY_COLOR
                    bgr = cv2.addWeighted(bgr, 1.0 - CAR_OVERLAY_ALPHA, c_overlay, CAR_OVERLAY_ALPHA, 0)
                    # Blue outline
                    c_arr = np.array(c_pts, dtype=np.int32)
                    bgr = cv2.polylines(bgr, [c_arr], True, CAR_OVERLAY_OUTLINE_COLOR, CAR_OVERLAY_THICKNESS)

        # --- write outputs ---
        lbl_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)

        label_path = lbl_dir / (image_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

        cv2.putText(
            bgr,
            f"{self.class_prefix}surface",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        if car_count:
            cv2.putText(
                bgr,
                f"car ({car_count})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        preview_path = preview_dir / image_path.name
        cv2.imwrite(str(preview_path), bgr)
        shutil.copy2(str(preview_path), str(annotated_dir / image_path.name))

        return {
            "label_path": str(label_path),
            "preview_path": str(preview_path),
            "annotated_path": str(annotated_dir / image_path.name),
            "surface_polys": 1 if yolo_poly else 0,
            "car_polys": car_count,
            "car_polys_detail": car_polys,
            "image_name": image_path.name,
            "width": w,
            "height": h,
            "pixel_points": pixel_pts,
            "area": area,
        }


def export_cvat_xml(
    results, output_path, task_name="Surface Auto-Annotator", class_prefix=CLASS_PREFIX,
    detect_cars=False,
):
    """Export annotation results to CVAT XML format.

    Args:
        results: list of dicts returned by Annotator.annotate()
        output_path: Path for the output .xml file
        task_name: name for the CVAT task
        class_prefix: prefix prepended to the surface class name
        detect_cars: when True, also include a "car" label and any car polygons
                      that were produced (clipped to surface, per-image).

    Returns:
        Path to the generated XML file
    """
    surface_name = f"{class_prefix}surface"
    output_path = Path(output_path)
    now = datetime.datetime.now().isoformat()

    label_lines = [
        f'      <label><name>{surface_name}</name><type>polygon</type><attributes/></label>',
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
        f"    <task><id>1</id><name>{task_name}</name><size>{len(results)}</size><mode>annotation</mode><created>{now}</created><updated>{now}</updated></task>",
        "    <labels>",
        *label_lines,
        "    </labels>",
        "  </meta>",
    ]

    for i, r in enumerate(results):
        lines.append(
            f'  <image id="{i}" name="{r["image_name"]}" width="{r["width"]}" height="{r["height"]}">'
        )
        if r["pixel_points"]:
            pts_str = ";".join(f"{x},{y}" for x, y in r["pixel_points"])
            lines.append(
                f'    <polygon label="{surface_name}" points="{pts_str}" z_order="0"/>'
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
