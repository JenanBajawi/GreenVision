"""
============================================================
  🌿 GreenVision — Plant Disease Detection
  Streamlit Web Application
  Umm Al-Qura University | Computer Vision Project
  Team Group 19 | Supervisor: Dr. Serene Noor Wali
============================================================
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import io
import urllib.request
import time

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

import urllib.request

MODEL_URLS = {
    "plant_model.pth": "https://huggingface.co/JenanB/greenvision-models/resolve/main/plant_model.pth",
    "severity_model.pth": "https://huggingface.co/JenanB/greenvision-models/resolve/main/severity_model.pth"
}

@st.cache_resource(show_spinner=False)
def download_models():
    for filename, url in MODEL_URLS.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename} (one-time setup)…"):
                urllib.request.urlretrieve(url, filename)
    return True

download_models()
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GreenVision · Botanical Diagnostics",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS — EDITORIAL BOTANICAL AESTHETIC
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;0,9..144,600;0,9..144,700;0,9..144,900;1,9..144,400&family=Manrope:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* ========== ROOT VARIABLES ========== */
    :root {
        --color-bg: #f5f1e8;
        --color-bg-cream: #faf6ed;
        --color-ink: #1a2e1a;
        --color-moss: #3d5a3d;
        --color-sage: #7a8d6f;
        --color-leaf: #4a6b3a;
        --color-gold: #b8924f;
        --color-rust: #a64b2a;
        --color-amber: #d4953c;
        --color-paper: #ffffff;
        --color-line: #d4cdb8;

        --font-display: 'Fraunces', Georgia, serif;
        --font-body: 'Manrope', system-ui, sans-serif;
    }

    /* ========== GLOBAL ========== */
    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--color-ink);
    }

    .stApp {
        background:
            radial-gradient(ellipse 800px 600px at top right, rgba(122, 141, 111, 0.08), transparent),
            radial-gradient(ellipse 600px 400px at bottom left, rgba(184, 146, 79, 0.06), transparent),
            var(--color-bg);
        background-attachment: fixed;
    }

    /* تخفي مكونات Streamlit الافتراضية */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4 {
        font-family: var(--font-display) !important;
        font-weight: 500 !important;
        color: var(--color-ink) !important;
        letter-spacing: -0.02em !important;
    }

    /* ========== MASTHEAD ========== */
    .masthead {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid var(--color-line);
        margin-bottom: 3rem;
        position: relative;
    }

    .masthead::before {
        content: "EST · 2026";
        position: absolute;
        top: 1rem;
        left: 0;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        color: var(--color-sage);
        font-weight: 500;
    }

    .masthead::after {
        content: "VOL · I";
        position: absolute;
        top: 1rem;
        right: 0;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        color: var(--color-sage);
        font-weight: 500;
    }

    .masthead-title {
        font-family: var(--font-display);
        font-size: clamp(3.5rem, 8vw, 6rem);
        font-weight: 400;
        line-height: 0.95;
        letter-spacing: -0.04em;
        margin: 0;
        color: var(--color-ink);
    }

    .masthead-title em {
        font-style: italic;
        font-weight: 300;
        color: var(--color-leaf);
    }

    .masthead-subtitle {
        font-family: var(--font-body);
        font-size: 0.75rem;
        letter-spacing: 0.4em;
        text-transform: uppercase;
        color: var(--color-sage);
        margin-top: 1rem;
        font-weight: 500;
    }

    .masthead-tagline {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 1.1rem;
        color: var(--color-moss);
        margin-top: 0.75rem;
        font-weight: 300;
    }

    /* ========== SECTION LABELS ========== */
    .section-label {
        font-size: 0.7rem;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        color: var(--color-gold);
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .section-label::before {
        content: "";
        width: 30px;
        height: 1px;
        background: var(--color-gold);
    }

    .section-title {
        font-family: var(--font-display);
        font-size: 2.5rem;
        font-weight: 400;
        margin: 0.5rem 0 1.5rem 0;
        color: var(--color-ink);
        line-height: 1.1;
    }

    /* ========== UPLOAD AREA ========== */
    .upload-container {
        background: var(--color-paper);
        border: 1px solid var(--color-line);
        border-radius: 4px;
        padding: 3rem 2rem;
        margin: 2rem 0;
        position: relative;
        box-shadow: 0 1px 3px rgba(26, 46, 26, 0.05);
    }

    .upload-container::before {
        content: "❋";
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--color-bg);
        color: var(--color-gold);
        padding: 0 1rem;
        font-size: 1rem;
    }

    /* تخصيص file uploader */
    [data-testid="stFileUploader"] {
        background: transparent;
    }

    [data-testid="stFileUploader"] section {
        background: var(--color-bg-cream) !important;
        border: 2px dashed var(--color-sage) !important;
        border-radius: 4px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--color-leaf) !important;
        background: var(--color-paper) !important;
    }

    [data-testid="stFileUploader"] button {
        background: var(--color-ink) !important;
        color: var(--color-bg-cream) !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        border-radius: 2px !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"] button:hover {
        background: var(--color-leaf) !important;
        transform: translateY(-1px);
    }

    /* ========== RESULT CARDS ========== */
    .specimen-card {
        background: var(--color-paper);
        border: 1px solid var(--color-line);
        padding: 2rem;
        position: relative;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(26, 46, 26, 0.06);
    }

    .specimen-card::before {
        content: "";
        position: absolute;
        top: 8px;
        left: 8px;
        right: 8px;
        bottom: 8px;
        border: 1px solid rgba(184, 146, 79, 0.2);
        pointer-events: none;
    }

    .specimen-label {
        font-size: 0.65rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--color-sage);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .specimen-name {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 2rem;
        font-weight: 400;
        color: var(--color-ink);
        margin: 0;
        line-height: 1.2;
    }

    .specimen-genus {
        font-family: var(--font-display);
        font-size: 0.9rem;
        color: var(--color-sage);
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }

    /* ========== METRICS ========== */
    .metric-block {
        text-align: center;
        padding: 1.5rem 1rem;
        border-left: 1px solid var(--color-line);
    }

    .metric-block:first-child {
        border-left: none;
    }

    .metric-label {
        font-size: 0.65rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        color: var(--color-sage);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-family: var(--font-display);
        font-size: 2.5rem;
        font-weight: 500;
        color: var(--color-ink);
        line-height: 1;
    }

    .metric-value.success { color: var(--color-leaf); }
    .metric-value.warning { color: var(--color-amber); }
    .metric-value.danger { color: var(--color-rust); }

    .metric-suffix {
        font-size: 1rem;
        color: var(--color-sage);
        font-weight: 300;
    }

    /* ========== STATUS BADGES ========== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        font-weight: 600;
        border-radius: 2px;
        margin-top: 1rem;
    }

    .status-healthy {
        background: rgba(74, 107, 58, 0.1);
        color: var(--color-leaf);
        border: 1px solid var(--color-leaf);
    }

    .status-mild {
        background: rgba(184, 146, 79, 0.1);
        color: var(--color-gold);
        border: 1px solid var(--color-gold);
    }

    .status-moderate {
        background: rgba(212, 149, 60, 0.1);
        color: var(--color-amber);
        border: 1px solid var(--color-amber);
    }

    .status-severe {
        background: rgba(166, 75, 42, 0.1);
        color: var(--color-rust);
        border: 1px solid var(--color-rust);
    }

    /* ========== DISEASE INFO ========== */
    .disease-description {
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-style: italic;
        line-height: 1.6;
        color: var(--color-moss);
        margin: 1.5rem 0;
        padding: 1rem 0;
        border-top: 1px solid var(--color-line);
        border-bottom: 1px solid var(--color-line);
    }

    /* ========== RECOMMENDATIONS ========== */
    .recommendation-item {
        display: flex;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 1px solid var(--color-line);
        align-items: flex-start;
    }

    .recommendation-item:last-child {
        border-bottom: none;
    }

    .recommendation-number {
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-style: italic;
        color: var(--color-gold);
        font-weight: 500;
        line-height: 1;
        min-width: 30px;
    }

    .recommendation-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--color-ink);
    }

    /* ========== PROBABILITY BARS ========== */
    .prob-row {
        margin-bottom: 1rem;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.4rem;
    }

    .prob-name {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 0.95rem;
        color: var(--color-ink);
    }

    .prob-value {
        font-family: var(--font-body);
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--color-moss);
        letter-spacing: 0.05em;
    }

    .prob-bar-bg {
        height: 4px;
        background: var(--color-line);
        position: relative;
        overflow: hidden;
        border-radius: 2px;
    }

    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--color-leaf), var(--color-gold));
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .prob-bar-fill.top {
        background: var(--color-leaf);
    }

    /* ========== INFO CARDS GRID ========== */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .info-card {
        background: var(--color-paper);
        border: 1px solid var(--color-line);
        padding: 2rem 1.5rem;
        position: relative;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(26, 46, 26, 0.08);
    }

    .info-card-number {
        font-family: var(--font-display);
        font-size: 0.75rem;
        font-style: italic;
        color: var(--color-gold);
        letter-spacing: 0.2em;
        margin-bottom: 1rem;
    }

    .info-card-title {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 1.4rem;
        color: var(--color-ink);
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .info-card-latin {
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--color-sage);
        margin-bottom: 1rem;
    }

    .info-card-desc {
        font-size: 0.85rem;
        line-height: 1.5;
        color: var(--color-moss);
    }

    /* ========== DIVIDER ========== */
    .ornamental-divider {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 3rem 0 2rem 0;
        color: var(--color-gold);
    }

    .ornamental-divider::before,
    .ornamental-divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--color-line), transparent);
    }

    .ornamental-divider span {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 1.2rem;
    }

    /* ========== FOOTER ========== */
    .colophon {
        margin-top: 5rem;
        padding-top: 2rem;
        border-top: 1px solid var(--color-line);
        text-align: center;
        font-size: 0.75rem;
        color: var(--color-sage);
        letter-spacing: 0.1em;
        line-height: 1.8;
    }

    .colophon strong {
        color: var(--color-moss);
        font-weight: 600;
    }

    /* ========== IMAGE FRAMES ========== */
    .image-frame {
        padding: 0.75rem;
        background: var(--color-paper);
        border: 1px solid var(--color-line);
        position: relative;
    }

    .image-frame::before {
        content: "";
        position: absolute;
        inset: 4px;
        border: 1px solid rgba(184, 146, 79, 0.15);
        pointer-events: none;
    }

    .image-caption {
        font-family: var(--font-display);
        font-style: italic;
        font-size: 0.85rem;
        color: var(--color-sage);
        text-align: center;
        margin-top: 0.75rem;
        letter-spacing: 0.05em;
    }

    .image-caption strong {
        color: var(--color-gold);
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-size: 0.7rem;
        font-style: normal;
    }

    /* ========== ANIMATIONS ========== */
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-up {
        animation: fadeUp 0.8s ease-out forwards;
    }

    .delay-1 { animation-delay: 0.1s; opacity: 0; }
    .delay-2 { animation-delay: 0.3s; opacity: 0; }
    .delay-3 { animation-delay: 0.5s; opacity: 0; }
    .delay-4 { animation-delay: 0.7s; opacity: 0; }

    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .masthead-title {
            font-size: 3rem;
        }
        .section-title {
            font-size: 1.8rem;
        }
        .specimen-name {
            font-size: 1.5rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DEVICE & CONSTANTS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4

CLASS_NAMES = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy"
]

# معلومات الأمراض بأسلوب نباتي علمي
DISEASE_INFO = {
    "Grape___Black_rot": {
        "common_name": "Black Rot",
        "latin_name": "Guignardia bidwellii",
        "description": "A fungal pathogen producing distinct circular lesions with darkened borders, often spreading rapidly under humid conditions.",
        "icon": "❋",
    },
    "Grape___Esca_(Black_Measles)": {
        "common_name": "Esca",
        "latin_name": "Phaeomoniella chlamydospora",
        "description": "A complex disease characterized by tiger-stripe patterns between leaf veins, often indicating advanced vascular infection.",
        "icon": "✦",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "common_name": "Leaf Blight",
        "latin_name": "Isariopsis griseola",
        "description": "Angular dark spots that progressively dry leaf tissue, leading to premature defoliation if untreated.",
        "icon": "✶",
    },
    "Grape___healthy": {
        "common_name": "Healthy Specimen",
        "latin_name": "Vitis vinifera",
        "description": "No pathological indicators detected. The specimen exhibits uniform pigmentation and intact structural integrity.",
        "icon": "✿",
    },
}

TREATMENTS = {
    "Grape___Black_rot": [
        "Apply systemic fungicide (Mancozeb or Myclobutanil) every 10–14 days during the growing season.",
        "Remove all infected leaves, mummified berries, and fallen debris immediately to break the disease cycle.",
        "Improve canopy air circulation through strategic pruning and trellising adjustments.",
        "Avoid overhead irrigation; switch to drip systems to keep foliage dry.",
    ],
    "Grape___Esca_(Black_Measles)": [
        "Remove and destroy severely infected vines — no curative chemical treatment exists for Esca complex.",
        "Disinfect pruning tools with 10% bleach solution between every cut to prevent spread.",
        "Avoid pruning during wet or humid weather when fungal spores are most active.",
        "Consider planting resistant varieties when replanting affected sections of the vineyard.",
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Apply copper-based fungicide preventatively, especially before periods of expected rainfall.",
        "Improve vineyard drainage and reduce standing water around vine bases.",
        "Remove and burn infected leaves to eliminate overwintering spore reservoirs.",
        "Monitor humidity levels closely; maintain canopy openness for rapid leaf drying.",
    ],
    "Grape___healthy": [
        "Continue weekly visual inspections for early disease detection during peak growing seasons.",
        "Maintain proper vine spacing (typically 1.5–2 m) and disciplined annual pruning schedules.",
        "Apply preventative fungicide treatments during humid periods as a precaution.",
        "Keep the vineyard floor free of weeds, fallen debris, and decomposing organic matter.",
    ],
}


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_classification_model(path="plant_model.pth"):
    """Load the trained ResNet34 classifier"""
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model = model.to(DEVICE).eval()
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    return None, False


@st.cache_resource(show_spinner=False)
def load_regression_model(path="severity_model.pth"):
    """Load the trained severity regression model"""
    model = models.resnet34(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model = model.to(DEVICE).eval()
            return model, True
        except Exception as e:
            return None, False
    return None, False


# ============================================================
# IMAGE PROCESSING
# ============================================================
def preprocess_image(img):
    """Prepare image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    rgb_img = np.array(img.resize((224, 224))) / 255.0
    return input_tensor, rgb_img


def classify(model, input_tensor):
    """Run classification and return prediction + probabilities"""
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    all_probs = {CLASS_NAMES[i]: probs[0][i].item() for i in range(NUM_CLASSES)}
    return pred, confidence, all_probs


def predict_severity(model, input_tensor):
    """Run regression for severity estimation"""
    if model is None:
        return None
    with torch.no_grad():
        output = model(input_tensor).item()
    return output * 100


def generate_gradcam(model, input_tensor, rgb_img, target_class):
    """Generate Grad-CAM heatmap"""
    if not GRADCAM_AVAILABLE:
        return None

    # تفعيل gradients مؤقتاً
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    try:
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(
            rgb_img.astype(np.float32),
            grayscale_cam,
            use_rgb=True
        )
        return visualization
    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return None


# ============================================================
# UI HELPERS
# ============================================================
def get_severity_status(severity, is_healthy):
    """Determine severity status class and label"""
    if is_healthy:
        return "healthy", "Healthy", "success"
    if severity is None:
        return "mild", "Detected", "warning"
    if severity < 15:
        return "mild", "Mild Infection", "success"
    elif severity < 35:
        return "moderate", "Moderate Infection", "warning"
    else:
        return "severe", "Severe Infection", "danger"


def render_probability_bar(name, value, is_top=False):
    """Render a single probability bar"""
    bar_class = "prob-bar-fill top" if is_top else "prob-bar-fill"
    return (
        f'<div class="prob-row">'
        f'<div class="prob-label">'
        f'<span class="prob-name">{name}</span>'
        f'<span class="prob-value">{value*100:.1f}%</span>'
        f'</div>'
        f'<div class="prob-bar-bg">'
        f'<div class="{bar_class}" style="width: {value*100}%;"></div>'
        f'</div>'
        f'</div>'
    )


# ============================================================
# MASTHEAD
# ============================================================
st.markdown("""
<div class="masthead fade-up">
    <h1 class="masthead-title">Green<em>Vision</em></h1>
    <div class="masthead-subtitle">Botanical Diagnostics · Computer Vision</div>
    <div class="masthead-tagline">A scientific instrument for early disease detection in <em>Vitis vinifera</em></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================
cls_model, cls_loaded = load_classification_model()
reg_model, reg_loaded = load_regression_model()

if not cls_loaded:
    st.error("""
    ⚠️ **Classification model not found.**

    Please ensure `plant_model.pth` is in the same directory as this app.
    """)
    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================
st.markdown("""
<div class="fade-up delay-1">
    <div class="section-label">Specimen Submission</div>
    <div class="section-title">Submit a <em>leaf specimen</em> for analysis</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop leaf image here or browse files",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# ANALYSIS RESULTS
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner(""):
        input_tensor, rgb_img = preprocess_image(image)

        # Run inference
        pred_idx, confidence, all_probs = classify(cls_model, input_tensor)
        predicted_class = CLASS_NAMES[pred_idx]
        disease_info = DISEASE_INFO[predicted_class]
        is_healthy = predicted_class == "Grape___healthy"

        # Severity (if model available)
        severity = predict_severity(reg_model, input_tensor) if reg_loaded else None
        if is_healthy and severity is not None:
            severity = max(0, severity - 5)  # تقليل عند الـ healthy

        # Grad-CAM
        gradcam_img = generate_gradcam(cls_model, input_tensor, rgb_img, pred_idx)

        # Brief artistic pause
        time.sleep(0.3)

    # === Image showcase ===
    st.markdown('<div class="ornamental-divider"><span>✦</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fade-up delay-1">
        <div class="section-label">Visual Examination</div>
    </div>
    """, unsafe_allow_html=True)

    col_orig, col_gradcam = st.columns(2)

    with col_orig:
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="image-caption">
            <strong>Plate I</strong><br>
            <em>Specimen as submitted</em>
        </div>
        """, unsafe_allow_html=True)

    with col_gradcam:
        if gradcam_img is not None:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(gradcam_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="image-caption">
                <strong>Plate II</strong><br>
                <em>Diagnostic activation map · regions of interest</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Grad-CAM visualization unavailable. Install `pytorch-grad-cam` to enable.")

    # === Diagnosis ===
    st.markdown('<div class="ornamental-divider"><span>❋</span></div>', unsafe_allow_html=True)

    status_class, status_label, value_class = get_severity_status(severity, is_healthy)

    st.markdown(f"""
    <div class="fade-up delay-2">
        <div class="section-label">Diagnostic Finding</div>
        <div class="specimen-card">
            <div class="specimen-label">Identification · Confidence {confidence*100:.1f}%</div>
            <h2 class="specimen-name">{disease_info['common_name']}</h2>
            <div class="specimen-genus">{disease_info['latin_name'].upper()}</div>
            <div class="status-badge status-{status_class}">
                <span>●</span> {status_label}
            </div>
            <div class="disease-description">
                {disease_info['description']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === Metrics ===
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(f"""
        <div class="metric-block">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{confidence*100:.1f}<span class="metric-suffix">%</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        if severity is not None:
            st.markdown(f"""
            <div class="metric-block">
                <div class="metric-label">Severity Index</div>
                <div class="metric-value {value_class}">{severity:.1f}<span class="metric-suffix">%</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-block">
                <div class="metric-label">Severity Index</div>
                <div class="metric-value" style="color: var(--color-sage);">—</div>
            </div>
            """, unsafe_allow_html=True)

    with col_m3:
        # عدد الفئات المحتملة بثقة > 5%
        likely_count = sum(1 for p in all_probs.values() if p > 0.05)
        st.markdown(f"""
        <div class="metric-block">
            <div class="metric-label">Likely Matches</div>
            <div class="metric-value">{likely_count}<span class="metric-suffix"> / 4</span></div>
        </div>
        """, unsafe_allow_html=True)

    # === Probability distribution ===
    st.markdown('<div class="ornamental-divider"><span>✦</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fade-up delay-3">
        <div class="section-label">Probability Distribution</div>
        <div class="section-title">Differential <em>diagnosis</em></div>
    </div>
    """, unsafe_allow_html=True)

    # ترتيب من الأعلى للأقل
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    prob_html = ""
    for i, (cls_name, prob) in enumerate(sorted_probs):
        info = DISEASE_INFO[cls_name]
        is_top = (i == 0)
        prob_html += render_probability_bar(info['common_name'], prob, is_top)

    st.markdown(f'<div class="specimen-card">{prob_html}</div>', unsafe_allow_html=True)

    # === Treatment recommendations ===
    st.markdown('<div class="ornamental-divider"><span>✶</span></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fade-up delay-4">
        <div class="section-label">Recommended Protocol</div>
        <div class="section-title">Treatment <em>regimen</em></div>
    </div>
    """, unsafe_allow_html=True)

    recommendations = TREATMENTS[predicted_class]
    rec_html = '<div class="specimen-card">'
    for i, rec in enumerate(recommendations, 1):
        rec_html += (
            f'<div class="recommendation-item">'
            f'<div class="recommendation-number">{i:02d}</div>'
            f'<div class="recommendation-text">{rec}</div>'
            f'</div>'
        )
    rec_html += '</div>'
    st.markdown(rec_html, unsafe_allow_html=True)


# ============================================================
# WELCOME STATE (no upload yet)
# ============================================================
else:
    st.markdown("""
    <div class="fade-up delay-2" style="text-align: center; padding: 2rem 0;">
        <p style="font-family: 'Fraunces', serif; font-style: italic; font-size: 1.2rem;
                  color: var(--color-moss); max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Upload an image of a grape leaf and our deep learning system will identify
            the disease, estimate its severity, and visualise the regions
            influencing the diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ornamental-divider"><span>❋ ✦ ❋</span></div>', unsafe_allow_html=True)

    # عرض الأمراض المدعومة
    st.markdown("""
    <div class="fade-up delay-3">
        <div class="section-label">Diagnostic Catalogue</div>
        <div class="section-title">Recognised <em>specimens</em></div>
    </div>
    """, unsafe_allow_html=True)

    cards_html = '<div class="info-grid">'
    for i, (key, info) in enumerate(DISEASE_INFO.items(), 1):
        cards_html += (
            f'<div class="info-card fade-up delay-{min(i, 4)}">'
            f'<div class="info-card-number">№ {i:02d} · {info["icon"]}</div>'
            f'<div class="info-card-title">{info["common_name"]}</div>'
            f'<div class="info-card-latin">{info["latin_name"]}</div>'
            f'<div class="info-card-desc">{info["description"]}</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # System capabilities
    st.markdown('<div class="ornamental-divider"><span>✦</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="fade-up">
        <div class="section-label">Methodology</div>
        <div class="section-title">A three-fold <em>analysis</em></div>
    </div>
    """, unsafe_allow_html=True)

    method_html = """
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-number">№ I · ❋</div>
            <div class="info-card-title">Classification</div>
            <div class="info-card-latin">ResNet34 · Transfer Learning</div>
            <div class="info-card-desc">
                A deep convolutional neural network trained on the PlantVillage corpus
                identifies the disease with 97.22% accuracy.
            </div>
        </div>
        <div class="info-card">
            <div class="info-card-number">№ II · ✦</div>
            <div class="info-card-title">Severity Estimation</div>
            <div class="info-card-latin">Regression Model</div>
            <div class="info-card-desc">
                A second neural network quantifies the proportion of affected leaf
                tissue as a continuous severity index.
            </div>
        </div>
        <div class="info-card">
            <div class="info-card-number">№ III · ✶</div>
            <div class="info-card-title">Visual Explanation</div>
            <div class="info-card-latin">Grad-CAM · XAI</div>
            <div class="info-card-desc">
                Gradient-weighted activation maps reveal precisely which leaf regions
                influenced the diagnostic decision.
            </div>
        </div>
    </div>
    """
    st.markdown(method_html, unsafe_allow_html=True)


# ============================================================
# COLOPHON / FOOTER
# ============================================================
st.markdown("""
<div class="colophon">
    <div><strong>GREENVISION</strong> · BOTANICAL DIAGNOSTICS INSTRUMENT</div>
    <div style="margin-top: 0.5rem;">
        UMM AL-QURA UNIVERSITY · COMPUTER VISION PROJECT · GROUP NINETEEN
    </div>
    <div style="margin-top: 0.5rem;">
        SUPERVISED BY DR. SERENE NOOR WALI · MMXXVI
    </div>
    <div style="margin-top: 1.5rem; font-family: 'Fraunces', serif; font-style: italic; color: var(--color-gold);">
        ❋ ✦ ❋
    </div>
</div>
""", unsafe_allow_html=True)
