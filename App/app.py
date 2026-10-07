import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm

from PIL import Image


# SETTINGS

MODEL_PATH = "best_final.keras"
IMG_SIZE   = 224

CLASSES = [
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
]

CLASS_COLORS = {
    "Covid-19":            "#E85D30",
    "Emphysema":           "#F59E0B",
    "Normal":              "#10B981",
    "Pneumonia-Bacterial": "#3B82F6",
    "Pneumonia-Viral":     "#8B5CF6",
}

CLASS_INFO = {
    "Covid-19":            "SARS-CoV-2 infection. Appears as bilateral ground-glass opacity.",
    "Emphysema":           "Chronic obstructive pulmonary disease. Lung hyperinflation.",
    "Normal":              "Normal chest X-ray. No pathology detected.",
    "Pneumonia-Bacterial": "Bacterial pneumonia. Focal consolidation.",
    "Pneumonia-Viral":     "Viral pneumonia. Diffuse interstitial infiltrates.",
}


# PAGE CONFIG

st.set_page_config(
    page_title="Pneumonia AI · DenseNet-121",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLE

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header   { visibility: hidden; }

.stApp { background: #0A0F1E; color: #E2E8F0; }

.header-bar {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #1E40AF33;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    color: #F8FAFC; margin: 0 0 0.3rem 0;
}
.header-sub { color: #94A3B8; font-size: 0.9rem; margin: 0; }

.result-card {
    border-radius: 16px; padding: 1.4rem 1.6rem;
    border: 1px solid; margin-bottom: 1rem;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem; font-weight: 700; margin: 0 0 0.3rem 0;
}
.result-desc { font-size: 0.88rem; color: #CBD5E1; margin: 0; line-height: 1.5; }

.info-card {
    background: #111827; border: 1px solid #1F2937;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.info-label {
    font-size: 0.75rem; color: #6B7280;
    text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 0.2rem 0;
}
.info-value { font-family: 'Space Mono', monospace; font-size: 1rem; color: #F1F5F9; margin: 0; }

.prob-row   { display: flex; align-items: center; gap: 10px; margin-bottom: 0.6rem; }
.prob-label { font-size: 0.82rem; color: #94A3B8; width: 160px; flex-shrink: 0; }
.prob-bar-bg { flex: 1; height: 8px; background: #1F2937; border-radius: 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; }
.prob-pct {
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    color: #E2E8F0; width: 48px; text-align: right; flex-shrink: 0;
}
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; font-family: 'Space Mono', monospace;
}
.warning-box {
    background: #1C1208; border: 1px solid #92400E44;
    border-radius: 10px; padding: 0.8rem 1rem; font-size: 0.82rem; color: #FCD34D;
}
.section-title {
    font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em; margin: 1.2rem 0 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)


# SESSION STATE

for key in ["preds", "pred_label", "pred_conf",
            "orig_img", "heatmap_img", "overlay_img", "overlay_bytes"]:
    if key not in st.session_state:
        st.session_state[key] = None


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)


# HELPERS

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


def make_smoothgrad(img_array, model, class_idx=None, n_samples=25, noise=0.12):
    img_tensor = tf.cast(img_array, tf.float32)

    preds = model(img_tensor, training=False)
    if class_idx is None:
        class_idx = int(tf.argmax(preds[0]))

    all_grads = []
    for _ in range(n_samples):
        noise_t = tf.random.normal(
            shape=tf.shape(img_tensor), mean=0.0, stddev=noise
        )
        noisy = tf.clip_by_value(img_tensor + noise_t, 0.0, 1.0)
        x_var = tf.Variable(noisy)

        with tf.GradientTape() as tape:
            p = model(x_var, training=False)
            s = p[:, class_idx]

        g = tape.gradient(s, x_var)
        if g is not None:
            all_grads.append(g[0].numpy())

    if not all_grads:
        return np.zeros((IMG_SIZE, IMG_SIZE)), class_idx, preds[0].numpy()

    mean_g  = np.mean(np.stack(all_grads, axis=0), axis=0)
    heatmap = np.max(np.abs(mean_g), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap, class_idx, preds[0].numpy()


def build_overlay(pil_img, heatmap, alpha=0.45):
    img_rgb = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heat_rs = cv2.resize(heatmap.astype(np.float32), (IMG_SIZE, IMG_SIZE))
    heat_rs = cv2.GaussianBlur(heat_rs, (15, 15), 0)
    heat_rs = heat_rs / (heat_rs.max() + 1e-8)
    colored = (mplcm.jet(heat_rs)[:, :, :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, colored, alpha, 0)
    return img_rgb, colored, overlay


def to_png_bytes(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf  = cv2.imencode(".png", img_bgr)
    return buf.tobytes()


def conf_badge(conf):
    if conf >= 0.85:
        return "🟢 High", "#10B981", "#052e16"
    elif conf >= 0.65:
        return "🟠 Medium", "#F59E0B", "#1c1008"
    else:
        return "🔴 Low", "#EF4444", "#1c0808"


# HEADER

st.markdown("""
<div class="header-bar">
    <p class="header-title">🫁 Pneumonia Detection AI</p>
    <p class="header-sub">DenseNet-121 · 5 classes · SmoothGrad Explainability · Bachelor's thesis</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:
    st.markdown('<p class="section-title">Model information</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <p class="info-label">Architecture</p>
        <p class="info-value">DenseNet-121</p>
    </div>
    <div class="info-card">
        <p class="info-label">Input size</p>
        <p class="info-value">224 × 224 px</p>
    </div>
    <div class="info-card">
        <p class="info-label">Explainability</p>
        <p class="info-value">SmoothGrad<br>
        <span style="font-size:0.8rem;color:#6B7280">25 samples · noise=0.12</span></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">Classes</p>', unsafe_allow_html=True)
    for cls in CLASSES:
        color = CLASS_COLORS[cls]
        st.markdown(
            f'<span class="badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color}44;margin-bottom:6px;display:inline-block">'
            f'{cls}</span><br>', unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
        ⚠️ For academic / demo use only.<br>
        Not intended for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"❌ Loading error:\n{model_error}")


# MAIN

if not model_loaded:
    st.error("The model was not loaded. Check MODEL_PATH in the settings.")
    st.stop()

uploaded = st.file_uploader(
    "Upload a chest x-ray",
    type=["png", "jpg", "jpeg"],
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem;">
        <p style="font-size:3rem;margin:0;">🫁</p>
        <p style="font-family:'Space Mono',monospace;font-size:1rem;color:#4B5563;">
            Upload a chest x-ray to start the analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

image_pil = Image.open(uploaded)

col_img, col_action = st.columns([1, 1])

with col_img:
    st.markdown('<p class="section-title">Radiography</p>', unsafe_allow_html=True)
    st.image(image_pil, width="stretch")

with col_action:
    st.markdown('<p class="section-title">Analysis</p>', unsafe_allow_html=True)

    btn_predict = st.button("🔍 Run Prediction", type="primary", use_container_width=True)
    btn_gradcam = st.button("🌡️ Create Heatmap",  use_container_width=True)

    # Prediction
    if btn_predict:
        with st.spinner("Analysing image..."):
            x     = preprocess(image_pil)
            preds = model.predict(x, verbose=0)[0]
            pidx  = int(np.argmax(preds))

            st.session_state.preds         = preds
            st.session_state.pred_label    = CLASSES[pidx]
            st.session_state.pred_conf     = float(preds[pidx])
            st.session_state.overlay_img   = None
            st.session_state.overlay_bytes = None

    # Show result
    if st.session_state.preds is not None:
        label = st.session_state.pred_label
        conf  = st.session_state.pred_conf
        preds = st.session_state.preds
        color = CLASS_COLORS[label]
        badge_text, badge_color, badge_bg = conf_badge(conf)

        st.markdown(f"""
        <div class="result-card" style="background:{color}11;border-color:{color}44;">
            <p class="result-label" style="color:{color};">{label}</p>
            <p style="margin:0 0 0.5rem 0;">
                <span class="badge" style="background:{badge_bg};color:{badge_color};">
                    {badge_text} &nbsp; {conf:.1%}
                </span>
            </p>
            <p class="result-desc">{CLASS_INFO[label]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Probability distribution</p>', unsafe_allow_html=True)
        for cls, p in zip(CLASSES, preds):
            c      = CLASS_COLORS[cls]
            width  = int(p * 100)
            bold   = "font-weight:600;" if cls == label else ""
            lcolor = "#F1F5F9" if cls == label else "#94A3B8"
            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-label" style="{bold}color:{lcolor}">{cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{width}%;background:{c};"></div>
                </div>
                <span class="prob-pct">{p:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    # SmoothGrad Heatmap
    if btn_gradcam:
        if st.session_state.preds is None:
            st.warning("Run the prediction first.")
        else:
            with st.spinner("Heatmap calculation (SmoothGrad, ~5 sec)..."):
                try:
                    x    = preprocess(image_pil)
                    pidx = int(np.argmax(st.session_state.preds))
                    heatmap, _, _ = make_smoothgrad(x, model, pidx)
                    orig, heat_img, overlay = build_overlay(image_pil, heatmap)

                    st.session_state.orig_img      = orig
                    st.session_state.heatmap_img   = heat_img
                    st.session_state.overlay_img   = overlay
                    st.session_state.overlay_bytes = to_png_bytes(overlay)
                except Exception as e:
                    st.error(f"Heatmap error: {e}")


# HEATMAP DISPLAY

if st.session_state.overlay_img is not None:
    st.markdown("---")
    st.markdown('<p class="section-title">SmoothGrad · Attention Analysis</p>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.image(st.session_state.orig_img,    caption="Original image",        width="stretch")
    with g2:
        st.image(st.session_state.heatmap_img, caption="Heatmap (SmoothGrad)",  width="stretch")
    with g3:
        st.image(st.session_state.overlay_img, caption="Overlay",               width="stretch")

    st.markdown("""
    <p style="font-size:0.82rem;color:#4B5563;margin-top:0.5rem;">
        🔴 Red = high attention &nbsp;|&nbsp;
        🔵 Blue = low attention &nbsp;|&nbsp;
        The model focuses on the highlighted areas for prediction.
    </p>
    """, unsafe_allow_html=True)

    st.download_button(
        label     = "⬇️ Download Heatmap",
        data      = st.session_state.overlay_bytes,
        file_name = f"heatmap_{st.session_state.pred_label}.png",
        mime      = "image/png",
    )