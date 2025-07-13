import math
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --------------------- Class Labels ---------------------
CLASS_NAMES = ["Bolt", "Bearing", "Nut", "Gear"]

# Optional: Info about each part
PART_INFO = {
    "Bolt": {"Description": "A bolt is a form of threaded fastener used with a nut."},
    "Bearing": {"Description": "A machine element that constrains motion and reduces friction."},
    "Nut": {"Description": "A nut is a type of fastener with a threaded hole."},
    "Gear": {"Description": "A rotating machine part with cut teeth that mesh with another toothed part."},
}

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.savedmodel", compile=False)

# --------------------- Preprocess Image ---------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# --------------------- Plot Prediction ---------------------
def plot_predictions(preds):
    preds = preds[0]
    bolt_score = preds[0] + preds[4]  # Merge Bolt + Screw
    bearing_score = preds[1]
    nut_score = preds[2]
    gear_score = preds[3]

    merged_preds = [bolt_score, bearing_score, nut_score, gear_score]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, merged_preds, color='skyblue', edgecolor='black')
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probabilities")
    for i, value in enumerate(merged_preds):
        ax.text(i, value + 0.02, f"{value:.2f}", ha='center')
    st.pyplot(fig)

# --------------------- Streamlit UI ---------------------
st.set_page_config("🛠️ Mechanical Part Identifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🛠️ Mechanical Part Identifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Upload an image to identify it as one of: Bolt, Bearing, Nut, or Gear. (Bolt includes Screw)</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------- Upload Section ---------------------
uploaded_file = st.file_uploader("📷 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300, caption="Preview")

    model = load_model()
    input_tensor = preprocess_image(image)
    predictions = model(input_tensor, training=False).numpy()

    # Merge predictions
    bolt_score = predictions[0][0] + predictions[0][4]
    bearing_score = predictions[0][1]
    nut_score = predictions[0][2]
    gear_score = predictions[0][3]
    merged_logits = tf.stack([bolt_score, bearing_score, nut_score, gear_score])
    predicted_index = tf.argmax(merged_logits).numpy()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(merged_logits[predicted_index])

    st.markdown("---")
    st.success(f"✅ Predicted Part: **{predicted_class}**")
    st.info(f"🔢 Confidence: **{confidence * 100:.2f}%**")

    if predicted_class in PART_INFO:
        st.markdown("### ℹ️ Part Description")
        for key, val in PART_INFO[predicted_class].items():
            st.markdown(f"**{key}:** {val}")

    plot_predictions(predictions)

# --------------------- Sample Images ---------------------
st.markdown("---")
st.markdown("### 🧪 Try with Sample Images")
st.markdown("Download and test with example images from the model's sample set.")

sample_dir = "samples"
if os.path.exists(sample_dir):
    sample_images = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if sample_images:
        images_per_row = 4
        total_rows = math.ceil(len(sample_images) / images_per_row)

        for row in range(total_rows):
            cols = st.columns(images_per_row)
            for i in range(images_per_row):
                img_index = row * images_per_row + i
                if img_index >= len(sample_images):
                    break

                filename = sample_images[img_index]
                filepath = os.path.join(sample_dir, filename)
                label = os.path.splitext(filename)[0].replace("_", " ").capitalize()

                with cols[i]:
                    st.image(filepath, caption=label, width=150)
                    with open(filepath, "rb") as file:
                        st.download_button(
                            label="⬇ Download",
                            data=file,
                            file_name=filename,
                            mime="image/jpeg",
                            key=f"download-{filename}"
                        )
    else:
        st.warning("⚠️ No image files found in the 'samples' folder.")
else:
    st.info("👆 Please upload an image to get started.")
