"""
Surface Auto-Annotator - CLI (road + walkway + bikepath, paved surface only)
---------------------------------------------------
Auto-labels images using SegFormer (Cityscapes).

- Input:  images/  (any jpg/png/jpeg)
- Output: labels/  (YOLO-seg .txt polygons)
- Preview: preview/ (images with red polygon outline, no fill overlay)
- Annotated copy: annotated images/
- CVAT export: cvat_annotations.xml

Classes:
  0  surface_surface  (road + walkway + bikepath merged into ONE polygon.
                 Cityscapes has no separate bikepath class, so paved bike paths
                 fall under road/sidewalk. Green vegetation is NOT included.)

Black regions (e.g., car bonnets) are excluded from annotations.
"""

from pathlib import Path
import shutil
import tempfile
from annotator_core import Annotator, CLASSES, export_cvat_xml, DUAL_MODE

# -------- Config --------
ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
PREVIEW_DIR = ROOT / "preview"
ANNOTATED_DIR = ROOT / "annotated images"
MODEL_DIR = ROOT / "models"
CVAT_XML_PATH = ROOT / "cvat_annotations.xml"
CLASS_PREFIX = "surface_"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    for d in [IMG_DIR, LBL_DIR, PREVIEW_DIR, ANNOTATED_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def write_classes_file():
    with open(ROOT / "classes.txt", "w") as f:
        f.write(f"{CLASS_PREFIX}surface\n")


def write_dataset_yaml():
    lines = [
        f"path: {ROOT.as_posix()}",
        "train: images",
        "val: images",
        "",
        "names:",
        f"  0: {CLASS_PREFIX}surface",
    ]
    (ROOT / "dataset.yaml").write_text("\n".join(lines))


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
    print("[+] Loading SegFormer (Cityscapes) for road + walkway...")
    annotator = Annotator(MODEL_DIR, class_prefix=CLASS_PREFIX, dual_mode=DUAL_MODE)

    summary_path = ROOT / "summary.csv"
    all_results = []
    with open(summary_path, "w") as f:
        f.write("image,surface_surface_polys,area_px\n")
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {img_path.name}")
            r = annotator.annotate(img_path, LBL_DIR, PREVIEW_DIR, ANNOTATED_DIR)
            all_results.append(r)
            f.write(f"{img_path.name},{r['surface_polys']},{r['area']}\n")

    # Export CVAT XML
    import_path = export_cvat_xml(
        all_results, CVAT_XML_PATH, class_prefix=CLASS_PREFIX
    )
    print(f"\n[OK] Done.")
    print(f"  Labels:        {LBL_DIR}")
    print(f"  Previews:      {PREVIEW_DIR}")
    print(f"  Annotated:     {ANNOTATED_DIR}")
    print(f"  Summary:       {summary_path}")
    print(f"  Classes:       {ROOT / 'classes.txt'}")
    print(f"  Dataset cfg:   {ROOT / 'dataset.yaml'}")
    print(f"  CVAT XML:      {import_path}")
    print("\nPreview images: red polygon outline = surface_surface (road + walkway + bikepath, paved only).")


if __name__ == "__main__":
    main()
