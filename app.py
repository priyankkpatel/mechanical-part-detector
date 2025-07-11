import math
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --------------------- Define Merged Class Labels ---------------------
CLASS_NAMES = ["Bolt", "Bearing", "Nut", "Gear"]  # Final displayed labels

# --------------------- Load SavedModel ---------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.savedmodel")

# --------------------- Preprocess Uploaded Image ---------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# --------------------- Plot Prediction Probabilities ---------------------
def plot_predictions(preds):
    preds = preds[0]  # shape: (5,) for 5 original classes

    # Merge Bolt (index 0) + Screw (index 4)
    bolt_score = preds[0] + preds[4]
    bearing_score = preds[1]
    nut_score = preds[2]
    gear_score = preds[3]

    merged_preds = [bolt_score, bearing_score, nut_score, gear_score]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(CLASS_NAMES, merged_preds, color='steelblue', edgecolor='black')
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("🔍 Prediction Probabilities", fontsize=14)
    for i, value in enumerate(merged_preds):
        ax.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10)
    st.pyplot(fig)

# --------------------- Streamlit UI ---------------------
st.set_page_config("🛠️ Mechanical Part Identifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🛠️ Mechanical Part Identifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:16px;'>Upload an image to identify it as one of: Bolt, Bearing, Nut, or Gear. (Bolt includes Screw)</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Upload image section
uploaded_file = st.file_uploader("📷 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.markdown("### 📸 Uploaded Image")
    st.image(image, width=300, caption="Preview", use_column_width=False)

    model = load_model()
    input_tensor = preprocess_image(image)
    predictions = model(input_tensor, training=False)  # ensure inference mode
    output_tensor = predictions.numpy()

    # Merge predictions
    bolt_score = output_tensor[0][0] + output_tensor[0][4]  # Bolt + Screw
    bearing_score = output_tensor[0][1]
    nut_score = output_tensor[0][2]
    gear_score = output_tensor[0][3]

    merged_logits = tf.stack([bolt_score, bearing_score, nut_score, gear_score])
    predicted_index = tf.argmax(merged_logits).numpy()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(merged_logits[predicted_index])

    st.markdown("---")
    st.success(f"### ✅ Predicted Part: **{predicted_class}**")
    st.info(f"🔢 Confidence: **{confidence * 100:.2f}%**")

    plot_predictions(output_tensor)

st.markdown("---")
st.markdown("### 🧪 Try with Sample Images")
st.markdown("<p style='font-size:15px;'>Download and test with example images from the model's sample set.</p>", unsafe_allow_html=True)

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
