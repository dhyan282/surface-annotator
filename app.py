"""
Surface Auto-Annotator - Web UI
-------------------------------
Run:  streamlit run app.py
Drag-and-drop images, hit Annotate.
Detects ONLY paved surface classes (road, walkway, bikepath) merged into a
single "surface_surface". No YOLO, no object detection, no vegetation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import shutil
import streamlit as st

MODEL_VARIANTS = {
    "b0 (fastest)": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b1":          "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "b2 (balanced)": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b3":          "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b4":          "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "b5 (max precision)": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
FAST_SIZE = 640
DUAL_MODE = False
CLASSES = [
    (0, "surface_surface"),
    (1, "car"),
]

try:
    import annotator_core as _core
    Annotator = _core.Annotator
    export_cvat_xml = getattr(_core, "export_cvat_xml", None)
    try:
        MODEL_VARIANTS = _core.MODEL_VARIANTS
    except AttributeError:
        pass
    try:
        FAST_SIZE = _core.FAST_SIZE
    except AttributeError:
        pass
    try:
        DUAL_MODE = _core.DUAL_MODE
    except AttributeError:
        pass
    try:
        CLASSES = _core.CLASSES
    except AttributeError:
        pass
except Exception as _imp_err:
    import traceback
    st.error(f"Failed to import annotator_core: {_imp_err}")
    st.code(traceback.format_exc())
    st.write("annotator_core loaded from:", getattr(_core, "__file__", "<unknown>") if "_core" in dir() else "<not loaded>")
    st.write("ROOT:", str(ROOT))
    st.stop()

IMG_DIR      = ROOT / "images"
LBL_DIR      = ROOT / "labels"
PREVIEW_DIR  = ROOT / "preview"
ANN_DIR      = ROOT / "annotated images"
MODEL_DIR    = ROOT / "models"

st.set_page_config(
    page_title="Surface Auto-Annotator",
    page_icon="ROAD",
    layout="wide",
)


@st.cache_resource
def load_annotator(model_name: str, tiled: bool, dual_mode: bool, detect_cars: bool, car_conf: float):
    return Annotator(
        MODEL_DIR,
        model_name=model_name,
        tiled=tiled,
        dual_mode=dual_mode,
        detect_cars=detect_cars,
        car_conf_threshold=car_conf,
    )


with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox(
        "Model size (speed vs precision)",
        list(MODEL_VARIANTS.keys()),
        index=list(MODEL_VARIANTS.keys()).index("b2 (balanced)"),
        help="Bigger models segment more precisely but run slower on CPU.",
    )
    mode_label = st.selectbox(
        "Inference mode",
        ["Balanced (single pass)", "High-res (tiled)"],
        help="Balanced resizes the photo to the model's training scale (fast, accurate). "
        "Tiled keeps native resolution for very high-res photos (slower).",
    )
    conf = st.slider(
        "Surface confidence threshold",
        0.10, 0.90, 0.40, 0.05,
        help="Lower = keep more pixels (fixes missed walkways); higher = stricter.",
    )
    green = st.toggle("Exclude green vegetation", value=True,
                       help="Removes grass/leaf pixels that are never paved surface.")
    fast = st.toggle("Fast mode (resize to 640px, b0 model)", value=True,
                       help="Resizes images to 640px before inference and uses the fastest model for much quicker annotation.")
    dual = st.toggle("Dual model (2nd SegFormer-B0)", value=DUAL_MODE,
                       help="Runs a second SegFormer model in parallel and merges their outputs for higher accuracy.")
    detect_cars = st.toggle("Detect cars (YOLOv8-seg)", value=False,
                              help="Also detect cars / trucks / buses and emit them as overlapping 'car' parts. "
                                   "Each car polygon is clipped to the surface mask, so only the part of the "
                                   "car that sits on the road / walkway is annotated (the part of the car that "
                                   "overlaps the surface).")
    car_conf = st.slider(
        "Car confidence threshold", 0.10, 0.90, 0.35, 0.05,
        help="Minimum YOLOv8-seg confidence to keep a car detection.",
        disabled=not detect_cars,
    )
    prefix = st.text_input(
        "Class name prefix",
        value="surface_",
        help="Prepended to the class name in all annotation outputs.",
    )
    st.divider()
    st.write("**Output folders** (auto-created):")
    st.code(
        f"Labels:     {LBL_DIR}\n"
        f"Previews:   {PREVIEW_DIR}\n"
        f"Annotated:  {ANN_DIR}\n"
        f"CVAT XML:   {ROOT / 'cvat_annotations.xml'}",
        language="text",
    )
    st.divider()
    st.write("**Classes detected**")
    st.write(f"`0` {prefix}surface")
    st.write("`1` car" + ("  (clipped to surface)" if detect_cars else "  *(disabled)*"))
    st.divider()
    st.write("Green overlay: off (red polygon outline only)")


st.title("Surface Auto-Annotator")
st.write(
    "Drag and drop road / walkway / bikepath photos. The model draws "
    "**red polygon outlines for surface** (road, walkway, and bikepath merged into "
    "one). Green vegetation is excluded."
)

uploaded = st.file_uploader(
    "Drop images here (jpg, jpeg, png, bmp, webp)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    go = st.button("Annotate", type="primary", disabled=not uploaded)
with col2:
    clear = st.button("Clear uploaded")
with col3:
    open_folder = st.button("Open annotated folder")

if clear:
    st.rerun()

if open_folder:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import os
        os.startfile(str(ANN_DIR))  # noqa
        st.toast(f"Opened {ANN_DIR}", icon="FOLDER")
    except Exception as e:
        st.warning(f"Folder: {ANN_DIR}  (open manually: {e})")

if uploaded and go:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    st.info("Loading model (first time for a new size may take ~30s)...")
    model_name = MODEL_VARIANTS[model_label]
    try:
        annotator = load_annotator(
            model_name,
            tiled=(mode_label == "High-res (tiled)"),
            dual_mode=dual,
            detect_cars=detect_cars,
            car_conf=car_conf,
        )
    except Exception as _annot_err:
        st.error("Model failed to load. Full error:")
        st.exception(_annot_err)
        st.stop()
    annotator.conf_threshold = conf
    annotator.green_exclude = green
    annotator.class_prefix = prefix
    annotator.fast_mode = fast
    annotator.fast_size = FAST_SIZE
    st.success("Model loaded.")

    progress = st.progress(0.0, text="Starting...")
    results_log = []

    for i, up in enumerate(uploaded, 1):
        ext = Path(up.name).suffix.lower() or ".jpg"
        dst = IMG_DIR / f"upload_{i:04d}{ext}"
        dst.write_bytes(up.getbuffer())

        progress.progress(
            (i - 1) / len(uploaded),
            text=f"Annotating {up.name} ({i}/{len(uploaded)})",
        )
        try:
            r = annotator.annotate(dst, LBL_DIR, PREVIEW_DIR, ANN_DIR)
            results_log.append((up.name, r))
        except Exception as e:
            st.error(f"Failed: {up.name} -- {e}")

    progress.progress(1.0, text="Done!")
    st.success(f"Annotated {len(results_log)} image(s). Saved to: {ANN_DIR}")

    # Export CVAT XML
    cvat_path = ROOT / "cvat_annotations.xml"
    if export_cvat_xml is None:
        st.warning("CVAT export unavailable: annotator_core has no export_cvat_xml")
    else:
        try:
            export_cvat_xml(
                [r for _, r in results_log], cvat_path, class_prefix=prefix,
                detect_cars=detect_cars,
            )
            st.success(f"CVAT XML: `{cvat_path}`")
        except Exception as e:
            st.warning(f"CVAT export failed: {e}")

    st.divider()
    st.subheader("Results")
    for name, r in results_log:
        st.write(
            f"**{name}** -- {prefix}surface: {r['surface_polys']} "
            f"({r['area']} px)  |  car: {r.get('car_polys', 0)}"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Annotated")
            st.image(str(r["preview_path"]), use_container_width=True)
        with c2:
            with open(r["preview_path"], "rb") as f:
                st.download_button(
                    f"Download {name}",
                    data=f.read(),
                    file_name=name,
                    mime="image/jpeg",
                    use_container_width=True,
                )
        st.divider()

elif uploaded and not go:
    st.info(f"{len(uploaded)} image(s) ready. Click **Annotate** to start.")
