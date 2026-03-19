import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinScan AI",
    page_icon="🔬",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.stApp {
    background-color: #0a0a0f;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.6rem;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff6b6b, #ff9f43, #ffd32a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #888;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
}

/* Upload box */
.upload-box {
    border: 1.5px dashed #333;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #111118;
    margin: 1rem 0;
}

/* Result cards */
.result-card {
    background: #111118;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 0.8rem 0;
}
.result-card.malignant {
    border-left: 4px solid #ff6b6b;
}
.result-card.benign {
    border-left: 4px solid #26de81;
}

.label-malignant {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #ff6b6b;
    font-weight: 700;
}
.label-benign {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #26de81;
    font-weight: 700;
}
.prob-text {
    color: #aaa;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

/* Progress bar override */
.stProgress > div > div {
    background: linear-gradient(90deg, #ff6b6b, #ff9f43);
    border-radius: 4px;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #444;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    font-family: 'Space Mono', monospace;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Image captions */
.img-caption {
    text-align: center;
    color: #666;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.3rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.warning-box {
    background: #1a1200;
    border: 1px solid #ff9f43;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #ff9f43;
    font-size: 0.82rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ARSkinCD_version_2.keras")
    return model

# ── GradCAM ─────────────────────────────────────────────────────────────────────
def gradcam(model, img_array):
    img_tensor = tf.Variable(tf.cast(img_array, tf.float32))

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        x = img_tensor
        x = model.get_layer('conv2d_3')(x)
        x = model.get_layer('max_pooling2d_3')(x)
        x = model.get_layer('conv2d_4')(x)
        x = model.get_layer('max_pooling2d_4')(x)
        conv_out = model.get_layer('conv2d_5')(x)
        x = model.get_layer('max_pooling2d_5')(conv_out)
        x = model.get_layer('flatten_1')(x)
        x = model.get_layer('dense_2')(x)
        preds = model.get_layer('dense_3')(x)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-8)
    return heatmap


def preprocess(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (128, 128))
    img_norm = img / 255.0
    return img, np.expand_dims(img_norm, axis=0)


def overlay_gradcam(orig_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_colored * 0.4 + orig_img * 0.6).astype(np.uint8)
    return heatmap_resized, overlay


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor='#0a0a0f')
    buf.seek(0)
    return Image.open(buf)


# ── UI ──────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔬 SkinScan AI</h1>
    <p>Melanoma detection with gradient-weighted class activation mapping</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a dermoscopy image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file)

    try:
        model = load_model()
    except Exception:
        st.error("⚠️ `clean_model.h5` not found. Place it in the same directory as `app.py`.")
        st.stop()

    with st.spinner("Analyzing..."):
        orig_img, img_input = preprocess(image)
        pred = model.predict(img_input, verbose=0)
        prob = float(pred[0][0])
        is_malignant = prob > 0.45

        heatmap = gradcam(model, img_input)
        heatmap_resized, overlay = overlay_gradcam(orig_img, heatmap)

    # ── Result card ────────────────────────────────────────────────────────────
    label = "MALIGNANT" if is_malignant else "BENIGN"
    card_class = "malignant" if is_malignant else "benign"
    label_class = "label-malignant" if is_malignant else "label-benign"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="{label_class}">{label}</div>
        <div class="prob-text">Cancerous probability: <b>{prob:.4f}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Confidence**")
    st.progress(prob)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── GradCAM visualization ──────────────────────────────────────────────────
    st.markdown("#### GradCAM Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(orig_img, use_container_width=True, clamp=True)
        st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor('#0a0a0f')
        ax.imshow(heatmap_resized, cmap='jet')
        ax.axis('off')
        st.image(fig_to_pil(fig), use_container_width=True)
        plt.close(fig)
        st.markdown('<div class="img-caption">Heatmap</div>', unsafe_allow_html=True)

    with col3:
        st.image(overlay, use_container_width=True, clamp=True)
        st.markdown('<div class="img-caption">GradCAM Overlay</div>', unsafe_allow_html=True)

    # ── Interpretation ─────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Interpretation")
    st.markdown("""
    - 🔴 **Red/Yellow** regions — high model attention (most influential for prediction)
    - 🔵 **Blue** regions — low model attention
    - GradCAM target layer: `conv2d_5` (last convolutional layer)
    """)

    st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and is not a substitute for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:2rem;">🩺</div>
        <div style="color:#555; margin-top:0.5rem; font-size:0.9rem;">Drop a dermoscopy image here or click to browse</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">SkinScan AI · Built with TensorFlow + Streamlit</div>', unsafe_allow_html=True)
