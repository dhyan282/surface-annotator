"""
Surface Auto-Annotator - CLI (single surface class)
---------------------------------------------------
Auto-labels images using SegFormer (Cityscapes) with single polygon.

- Input:  images/  (any jpg/png/jpeg)
- Output: labels/  (YOLO-seg .txt polygons, one line per polygon:
                     "<class_id> <x1> <y1> <x2> <y2> ...")
- Preview: preview/ (images with class-colored polygon outline + filled overlay)
- Annotated copy: annotated images/
- CVAT export: cvat_annotations.xml

Classes (single merged surface class):
  0  surface_road       - all paved surfaces merged (road, walkway, bikepath)
"""

from pathlib import Path
import shutil
import tempfile
from annotator_core import (
    Annotator, CLASSES, export_cvat_xml, DUAL_MODE, CLASS_PREFIX,
    SURFACE_SURFACE_NAME,
)

# -------- Config --------
ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
PREVIEW_DIR = ROOT / "preview"
ANNOTATED_DIR = ROOT / "annotated images"
MODEL_DIR = ROOT / "models"
CVAT_XML_PATH = ROOT / "cvat_annotations.xml"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    for d in [IMG_DIR, LBL_DIR, PREVIEW_DIR, ANNOTATED_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def write_classes_file():
    """One line per (class_id, class_name) pair in CLASSES."""
    lines = [f"{cid} {name}" for cid, name in CLASSES]
    (ROOT / "classes.txt").write_text("\n".join(lines) + "\n")


def write_dataset_yaml():
    """YOLO dataset config with the per-class name map."""
    lines = [
        f"path: {ROOT.as_posix()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]
    for cid, name in CLASSES:
        lines.append(f"  {cid}: {name}")
    (ROOT / "dataset.yaml").write_text("\n".join(lines) + "\n")


def main():
    ensure_dirs()
    write_classes_file()
    write_dataset_yaml()

    images = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        print(f"\n[!] No images found in: {IMG_DIR}")
        print("    Drop some .jpg / .png files in there and re-run.\n")
        return

    print(f"[+] Found {len(images)} images.")
    print("[+] Loading SegFormer (Cityscapes) for road + walkway + bikepath...")
    annotator = Annotator(
        MODEL_DIR,
        class_prefix=CLASS_PREFIX,
        dual_mode=DUAL_MODE,
    )

    summary_path = ROOT / "summary.csv"
    all_results = []
    with open(summary_path, "w") as f:
        f.write("image,road_polys,area_px\n")
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")
            r = annotator.annotate(img_path, LBL_DIR, PREVIEW_DIR, ANNOTATED_DIR)
            all_results.append(r)
            total_area = sum(
                p.get("area", 0) for p in r.get("polygons", [])
            )
            f.write(
                f"{img_path.name},"
                f"{r.get('road_polys', 0)},"
                f"{total_area:.1f}\n"
            )

    # Export CVAT XML
    import_path = export_cvat_xml(
        all_results, CVAT_XML_PATH, class_prefix=CLASS_PREFIX, detect_cars=False,
    )
    print(f"\n[OK] Done.")
    print(f"  Labels:        {LBL_DIR}")
    print(f"  Previews:      {PREVIEW_DIR}")
    print(f"  Annotated:     {ANNOTATED_DIR}")
    print(f"  Summary:       {summary_path}")
    print(f"  Classes:       {ROOT / 'classes.txt'}")
    print(f"  Dataset cfg:   {ROOT / 'dataset.yaml'}")
    print(f"  CVAT XML:      {import_path}")
    print(
        "\nPreview images: red outline = " + CLASS_PREFIX + "road"
    )


if __name__ == "__main__":
    main()
