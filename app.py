"""Streamlit review dashboard for the MedGuard PathMNIST pilot model."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from medmnist import INFO, PathMNIST
from scripts.blur import KERNEL_SIZES, apply_blur
from scripts.train_baseline import DATA_ROOT, PROJECT_ROOT, SmallCNN


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "baseline" / "best_model.pt"
UNCERTAINTY_METRICS_PATH = PROJECT_ROOT / "outputs" / "uncertainty" / "metrics.json"
CLASS_NAMES = [INFO["pathmnist"]["label"][str(index)] for index in range(9)]
TO_TENSOR = transforms.ToTensor()

st.set_page_config(page_title="MedGuard | Patch review", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
      :root { --ink:#172b4d; --muted:#65758b; --line:#e3eaf2; --blue:#2563eb; --mint:#dff7f1; }
      html, body, [class*="css"] { font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
      [data-testid="stAppViewContainer"] { background:#f4f7fb; }
      [data-testid="stHeader"] { background:rgba(244,247,251,.92); }
      [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
      .block-container { max-width:1320px; padding-top:1.5rem; padding-bottom:3rem; }
      [data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4 { color:var(--ink) !important; font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; letter-spacing:-.025em; }
      [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stMetricValue"] { color:var(--ink); }
      [data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color:var(--ink); font-weight:600; }
      .hero { background:linear-gradient(115deg,#11294b 0%,#174c70 58%,#137e82 100%); padding:26px 30px; border-radius:20px; margin:0 0 20px; color:white; box-shadow:0 12px 28px rgba(23,58,91,.15); }
      .hero-row { display:flex; align-items:center; justify-content:space-between; gap:20px; }
      .hero h1 { color:#fff !important; margin:0 0 6px; font-size:2rem; }
      .hero p { color:#d7e7f3; margin:0; font-size:1rem; }
      .hero-tag { border:1px solid rgba(255,255,255,.3); border-radius:999px; padding:7px 12px; color:#e6f8f6; background:rgba(255,255,255,.09); font-size:.78rem; font-weight:700; letter-spacing:.06em; white-space:nowrap; }
      .section-kicker { color:#58708b; font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin:4px 0 8px; }
      .helper { color:var(--muted); font-size:.9rem; }
      div[data-testid="stAlert"] { border-radius:12px; }
      @media (max-width:700px) { .hero { padding:22px; } .hero-row { align-items:flex-start; flex-direction:column; } .hero h1 { font-size:1.7rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> tuple[SmallCNN, torch.device, float]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("Baseline checkpoint not found. Run scripts/train_baseline.py first.")
    if not UNCERTAINTY_METRICS_PATH.is_file():
        raise FileNotFoundError("Review threshold missing. Run scripts/evaluate_uncertainty.py first.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    metrics = json.loads(UNCERTAINTY_METRICS_PATH.read_text())
    threshold = float(metrics["validation"]["confidence_threshold"])
    return model, device, threshold


@st.cache_resource
def load_test_samples() -> tuple[PathMNIST, dict[str, list[int]]]:
    dataset = PathMNIST(split="test", root=str(DATA_ROOT), download=False)
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    by_class = {CLASS_NAMES[class_id]: np.flatnonzero(labels == class_id).tolist() for class_id in range(9)}
    return dataset, by_class


def normalize_image(image: Image.Image) -> Image.Image:
    """Composite transparency on white and return RGB."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def predict(image_tensor: torch.Tensor, model: SmallCNN, device: torch.device) -> tuple[int, float, np.ndarray]:
    with torch.inference_mode():
        probabilities = model(image_tensor.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu().numpy()
    class_id = int(probabilities.argmax())
    return class_id, float(probabilities[class_id]), probabilities


def show_result(title: str, class_id: int, confidence: float, threshold: float) -> None:
    st.markdown(f"#### {title}")
    st.caption("Predicted class")
    st.markdown(f"**{CLASS_NAMES[class_id]}**")
    st.metric("Model confidence", f"{confidence:.1%}")
    if confidence >= threshold:
        st.success("Accepted by the review threshold")
    else:
        st.warning("Review recommended · confidence is below the validation threshold")


try:
    model, device, confidence_threshold = load_predictor()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

st.markdown(
    """
    <div class="hero"><div class="hero-row">
      <div><h1>MedGuard</h1><p>Image-patch review with confidence-aware routing</p></div>
    </div></div>
    """,
    unsafe_allow_html=True,
)

try:
    test_dataset, examples_by_class = load_test_samples()
except (FileNotFoundError, RuntimeError) as error:
    st.error(f"Could not load PathMNIST examples: {error}")
    st.stop()

with st.container(border=True):
    # st.markdown('<div class="section-kicker">Image review</div>', unsafe_allow_html=True)
    sample_class = st.selectbox("Samples", CLASS_NAMES, label_visibility="visible")
    sample_indices = examples_by_class[sample_class][:20]
    selected_option = st.selectbox(
        "Example patch",
        range(len(sample_indices)),
        format_func=lambda option: f"Example {option + 1} · test image {sample_indices[option]}",
        label_visibility="collapsed",
    )
    sample_index = sample_indices[selected_option]
    input_image = normalize_image(Image.fromarray(np.asarray(test_dataset.imgs[sample_index])))
    dataset_label = sample_class
    source_description = f"PathMNIST test example · dataset label: {dataset_label}"

    prepared_image = input_image.resize((28, 28), Image.Resampling.LANCZOS)
    clean_tensor = TO_TENSOR(prepared_image)
    clean_class, clean_confidence, clean_probabilities = predict(clean_tensor, model, device)

    input_col, analysis_col = st.columns([0.9, 1.35], gap="large")
    with input_col:
        st.markdown("#### Input patch")
        st.image(input_image, width="stretch")
        if dataset_label:
            st.caption(f"Reference dataset label: **{dataset_label}**")
        if input_image.size != (28, 28):
            st.caption("Resized to 28 × 28 pixels to match model training input.")

    with analysis_col:
        st.markdown("#### Robustness check")
        control_col, severity_col = st.columns([1, 1.3])
        with control_col:
            blur_type = st.selectbox("Stress type", ["Defocus-style", "Motion"])
        with severity_col:
            blur_severity = st.slider(
                "Blur severity", min_value=0, max_value=5, value=0,
                help="0 leaves the patch unchanged. Levels 1–5 use the synthetic kernels from the stress evaluation.",
            )

        if blur_severity == 0:
            st.image(prepared_image, caption="Model input · 28 × 28", width=220)
            show_result("Prediction", clean_class, clean_confidence, confidence_threshold)
            probabilities = clean_probabilities
        else:
            corruption = "defocus" if blur_type == "Defocus-style" else "motion"
            kernel_size = KERNEL_SIZES[blur_severity - 1]
            blurred_tensor = apply_blur(clean_tensor.unsqueeze(0).to(device), corruption, kernel_size)[0].cpu()
            blurred_image = transforms.ToPILImage()(blurred_tensor.clamp(0, 1))
            result_original, result_blurred = st.columns(2, gap="medium")
            with result_original:
                st.image(prepared_image, caption="Original model input", width=220)
                show_result("Original", clean_class, clean_confidence, confidence_threshold)
            with result_blurred:
                st.image(blurred_image, caption=f"{blur_type} · level {blur_severity}", width=220)
                blur_class, blur_confidence, probabilities = predict(blurred_tensor, model, device)
                show_result("Under stress", blur_class, blur_confidence, confidence_threshold)

    st.markdown("---")
    st.markdown("#### Class probability profile")
    st.bar_chart(
        {CLASS_NAMES[index]: float(probabilities[index]) for index in range(9)},
        horizontal=True,
        x_label="Probability",
        y_label="Tissue class",
    )
    st.caption("Built-in examples come from the held-out test split; their labels are shown only for comparison and are never used by the model.")
