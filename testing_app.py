import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# page config
st.set_page_config(
    page_title="Skin Cancer Detector",
    page_icon="🧬",
    layout="centered"
)

# custom styling
st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# load model
model = tf.keras.models.load_model("ARSkinCD_version_2.keras")

IMG_SIZE = 128

st.markdown('<p class="big-title">🧬 Skin Cancer Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a dermoscopic skin lesion image</p>', unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_SIZE,IMG_SIZE))
    img = np.array(img)/255
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    with col2:

        st.subheader("Prediction Result")

        if prediction > 0.4:
            st.error("⚠️ Malignant")
        else:
            st.success("✅ Benign")

        confidence = float(prediction)

        st.write("Confidence Score")
        st.progress(confidence)

        st.write(f"{confidence*100:.2f}% probability")

st.write("")
st.caption("⚠️ This tool is for research purposes only and not a medical diagnosis.")