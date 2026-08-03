"""
Surface Auto-Annotator - Web UI
-------------------------------
Run:  streamlit run app.py
Drag-and-drop images, hit Annotate.
Detects paved surface classes (road, walkway, bikepath) and -- optionally --
cars. Each class is emitted as its own polygon for precise annotation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import shutil
import streamlit as st

# Reasonable defaults for the UI; the canonical config lives in
# annotator_core.py and we import those constants below.
try:
    import annotator_core as _core
    Annotator = _core.Annotator
    export_cvat_xml = getattr(_core, "export_cvat_xml", None)
    MODEL_VARIANTS = _core.MODEL_VARIANTS
    FAST_SIZE = _core.FAST_SIZE
    DUAL_MODE = _core.DUAL_MODE
    CLASSES = _core.CLASSES
    CLASS_PREFIX = _core.CLASS_PREFIX
    OVERLAY_COLORS_BGR = _core.OVERLAY_COLORS_BGR
except Exception as _imp_err:
    import traceback
    st.error(f"Failed to import annotator_core: {_imp_err}")
    st.code(traceback.format_exc())
    st.write(
        "annotator_core loaded from:",
        getattr(_core, "__file__", "<unknown>") if "_core" in dir() else "<not loaded>",
    )
    st.write("ROOT:", str(ROOT))
    st.stop()

IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
PREVIEW_DIR = ROOT / "preview"
ANN_DIR = ROOT / "annotated images"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Surface Auto-Annotator",
    page_icon=":material/straighten:",
    layout="wide",
)


@st.cache_resource
def load_annotator(
    model_name: str,
    tiled: bool,
    dual_mode: bool,
    detect_cars: bool,
    car_conf: float,
    fast_mode: bool,
    multi_scale: bool,
    tta: bool,
    detect_bikepath: bool,
):
    return Annotator(
        MODEL_DIR,
        model_name=model_name,
        tiled=tiled,
        dual_mode=dual_mode,
        detect_cars=detect_cars,
        car_conf_threshold=car_conf,
        fast_mode=fast_mode,
        multi_scale=multi_scale,
        tta=tta,
        detect_bikepath=detect_bikepath,
    )


# ---- Sidebar ----
with st.sidebar:
    st.header("Model")
    model_label = st.selectbox(
        "Model size (speed vs precision)",
        list(MODEL_VARIANTS.keys()),
        index=list(MODEL_VARIANTS.keys()).index("b3"),
        help="Bigger models segment more precisely but run slower on CPU.",
    )
    mode_label = st.selectbox(
        "Inference mode",
        ["Balanced (single pass)", "High-res (tiled)"],
        help="Balanced resizes the photo to the model's training scale (fast, accurate). "
        "Tiled keeps native resolution for very high-res photos (slower).",
    )

    st.header("Precision")
    tta = st.toggle(
        "Test-time augmentation (TTA)",
        value=True,
        help="Run the model on the original AND a horizontally-flipped image, then "
        "average. Roads and walkways are left-right symmetric, so this is a free "
        "precision boost at ~2x runtime.",
    )
    multi_scale = st.toggle(
        "Multi-scale inference",
        value=False,
        help="Run the model at 512 / 768 / 1024 px and average the probabilities. "
        "Great for variable-resolution photos. ~3x runtime.",
    )
    fast = st.toggle(
        "Fast mode (resize to 768px)",
        value=False,
        help="Resizes images to 768px before inference. Faster but less precise; "
        "leave off for max precision.",
    )
    detect_bikepath = st.toggle(
        "Detect bike path (sub-class of road)",
        value=True,
        help="Cityscapes has no separate bike-path class. When this is on, the "
        "annotator looks for a red-painted / curb-bordered strip inside the road "
        "mask and emits it as its own surface_bikepath polygon (class 2).",
    )

    st.header("Classes")
    conf = st.slider(
        "Surface confidence threshold",
        0.10, 0.90, 0.40, 0.05,
        help="Lower = keep more pixels (fixes missed walkways); higher = stricter.",
    )
    green = st.toggle(
        "Exclude green vegetation",
        value=True,
        help="Removes grass/leaf pixels that are never paved surface.",
    )
    prefix = st.text_input(
        "Class name prefix",
        value=CLASS_PREFIX,
        help="Prepended to the class names in all annotation outputs.",
    )

    st.header("Advanced")
    dual = st.toggle(
        "Dual model (2nd SegFormer-B0)",
        value=DUAL_MODE,
        help="Runs a second SegFormer model in parallel and merges their outputs for "
        "higher accuracy.",
    )
    detect_cars = st.toggle(
        "Detect cars (YOLOv8-seg)",
        value=False,
        help="Also detect cars / trucks / buses and emit them as 'car' parts. Each "
        "car polygon is clipped to the surface mask, so only the part of the car "
        "that sits on the road / walkway is annotated.",
    )
    car_conf = st.slider(
        "Car confidence threshold",
        0.10, 0.90, 0.35, 0.05,
        help="Minimum YOLOv8-seg confidence to keep a car detection.",
        disabled=not detect_cars,
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
    st.write("`0` " + prefix + "road  *(red)*")
    st.write("`1` " + prefix + "walkway  *(green)*")
    st.write("`2` " + prefix + "bikepath  *(cyan, when detected)*")
    st.write("`3` car" + ("  *(blue, clipped to surface)*" if detect_cars else "  *(disabled)*"))
    st.divider()
    st.write("Each class is drawn with its own outline color in the preview.")


# ---- Main page ----
st.title("Surface Auto-Annotator")
st.write(
    "Drag and drop road / walkway / bikepath photos. The annotator emits "
    "**per-class polygons**: " + prefix + "road, " + prefix + "walkway, "
    "and " + prefix + "bikepath. Green vegetation is excluded. Optionally "
    "annotates overlapping cars as a separate class."
)

uploaded = st.file_uploader(
    "Drop images here (jpg, jpeg, png, bmp, webp)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    go = st.button(
        "Annotate", type="primary", disabled=not uploaded,
        icon=":material/auto_awesome:",
    )
with col2:
    clear = st.button("Clear uploaded", icon=":material/refresh:")
with col3:
    open_folder = st.button(
        "Open annotated folder", icon=":material/folder_open:"
    )

if clear:
    st.rerun()

if open_folder:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import os
        os.startfile(str(ANN_DIR))  # noqa
        st.toast(f"Opened {ANN_DIR}", icon=":material/check_circle:")
    except Exception as e:
        st.warning(f"Folder: {ANN_DIR}  (open manually: {e})")


def class_color_swatch_bgr(bgr):
    b, g, r = bgr
    return f"rgb({r},{g},{b})"


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
            fast_mode=fast,
            multi_scale=multi_scale,
            tta=tta,
            detect_bikepath=detect_bikepath,
        )
    except Exception as _annot_err:
        st.error("Model failed to load. Full error:")
        st.exception(_annot_err)
        st.stop()
    annotator.conf_threshold = conf
    annotator.green_exclude = green
    annotator.class_prefix = prefix
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

    # ---- CVAT XML ----
    cvat_path = ROOT / "cvat_annotations.xml"
    if export_cvat_xml is None:
        st.warning("CVAT export unavailable: annotator_core has no export_cvat_xml")
    else:
        try:
            export_cvat_xml(
                [r for _, r in results_log],
                cvat_path,
                class_prefix=prefix,
                detect_cars=detect_cars,
            )
            st.success(f"CVAT XML: `{cvat_path}`")
        except Exception as e:
            st.warning(f"CVAT export failed: {e}")

    # ---- Summary table ----
    st.divider()
    st.subheader("Summary")
    summary_cols = st.columns(4)
    totals = {"road": 0, "walkway": 0, "bike": 0, "car": 0}
    for name, r in results_log:
        totals["road"] += r.get("road_polys", 0)
        totals["walkway"] += r.get("walkway_polys", 0)
        totals["bike"] += r.get("bike_polys", 0)
        totals["car"] += r.get("car_polys", 0)
    summary_cols[0].metric(f"{prefix}road", totals["road"])
    summary_cols[1].metric(f"{prefix}walkway", totals["walkway"])
    summary_cols[2].metric(f"{prefix}bikepath", totals["bike"])
    summary_cols[3].metric("car", totals["car"])

    # ---- Per-image results ----
    st.divider()
    st.subheader("Per-image results")
    for name, r in results_log:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"**{name}**  &nbsp; "
                f"<span style='color:{class_color_swatch_bgr(OVERLAY_COLORS_BGR[0])}'>"
                f":material/straighten: {prefix}road {r.get('road_polys', 0)}</span> &nbsp; "
                f"<span style='color:{class_color_swatch_bgr(OVERLAY_COLORS_BGR[1])}'>"
                f":material/directions_walk: {prefix}walkway {r.get('walkway_polys', 0)}</span> &nbsp; "
                f"<span style='color:{class_color_swatch_bgr(OVERLAY_COLORS_BGR[2])}'>"
                f":material/directions_bike: {prefix}bikepath {r.get('bike_polys', 0)}</span> &nbsp; "
                f"<span style='color:{class_color_swatch_bgr(OVERLAY_COLORS_BGR[3])}'>"
                f":material/directions_car: car {r.get('car_polys', 0)}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Annotated preview")
            st.image(str(r["preview_path"]))
        with c2:
            st.caption("Polygon breakdown")
            poly_rows = [
                {
                    "class": f"{prefix}road",
                    "polys": r.get("road_polys", 0),
                    "approx. pixels": r.get("polygons", [{}])[0].get("area", 0)
                    if r.get("road_polys", 0)
                    else 0,
                },
                {
                    "class": f"{prefix}walkway",
                    "polys": r.get("walkway_polys", 0),
                    "approx. pixels": 0,
                },
                {
                    "class": f"{prefix}bikepath",
                    "polys": r.get("bike_polys", 0),
                    "approx. pixels": 0,
                },
                {
                    "class": "car",
                    "polys": r.get("car_polys", 0),
                    "approx. pixels": 0,
                },
            ]
            st.dataframe(poly_rows, hide_index=True, width="stretch")
            with open(r["preview_path"], "rb") as f:
                st.download_button(
                    f"Download {name}",
                    data=f.read(),
                    file_name=name,
                    mime="image/jpeg",
                    icon=":material/download:",
                    width="stretch",
                )
        st.divider()

elif uploaded and not go:
    st.info(f"{len(uploaded)} image(s) ready. Click **Annotate** to start.")
