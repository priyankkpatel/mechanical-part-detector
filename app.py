import math
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --------------------- Define Labels and Info ---------------------
CLASS_NAMES = ["Bolt", "Bearing", "Nut", "Gear"]

PART_INFO = {
    "Bolt": {"Material": "Steel", "Use": "Fastening", "Avg Price": "$0.10"},
    "Bearing": {"Material": "Chromium Steel", "Use": "Reduce Friction", "Avg Price": "$5"},
    "Nut": {"Material": "Brass", "Use": "Threaded Fastening", "Avg Price": "$0.05"},
    "Gear": {"Material": "Alloy Steel", "Use": "Torque Transmission", "Avg Price": "$10"},
}

# --------------------- Load Model ---------------------
@st.cache_resource
def load_model():
    try:
        # Option 1: Load SavedModel directly (recommended)
        model = tf.keras.models.load_model("model.savedmodel")
        return model
    except Exception as e1:
        try:
            # Option 2: Load as SavedModel with tf.saved_model
            model = tf.saved_model.load("model.savedmodel")
            return model
        except Exception as e2:
            st.error(f"Failed to load model: {e1}, {e2}")
            return None

# --------------------- Preprocess Image ---------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)

# --------------------- Make Prediction ---------------------
def make_prediction(model, input_tensor):
    """Handle prediction for both Keras model and SavedModel"""
    try:
        # Try Keras model predict method
        predictions = model.predict(input_tensor, verbose=0)
        return predictions
    except AttributeError:
        # Handle SavedModel loaded with tf.saved_model.load
        try:
            # Try different signature keys
            infer = model.signatures["serving_default"]
            predictions = infer(tf.constant(input_tensor))
            # Extract the output (might be in different formats)
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[0]
            return predictions.numpy()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None

# --------------------- Top 3 Predictions Table ---------------------
def show_top_predictions(preds):
    if preds is None:
        return
    
    # Handle different prediction formats
    if len(preds.shape) > 1:
        preds = preds[0]
    
    # Check if we have 5 classes (including the extra bolt class)
    if len(preds) == 5:
        bolt_score = preds[0] + preds[4]
        scores = {
            "Bolt": bolt_score,
            "Bearing": preds[1],
            "Nut": preds[2],
            "Gear": preds[3]
        }
    else:
        # Handle 4 classes directly
        scores = {name: score for name, score in zip(CLASS_NAMES, preds)}
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    st.markdown("### 🧠 Top 3 Predictions")
    st.table({label: f"{score*100:.2f}%" for label, score in sorted_scores[:3]})

# --------------------- Show Part Info ---------------------
def show_part_info(part_name):
    st.sidebar.markdown("### 🧾 Part Information")
    info = PART_INFO.get(part_name, {})
    for key, val in info.items():
        st.sidebar.write(f"**{key}**: {val}")

# --------------------- Plot Probabilities ---------------------
def plot_predictions(preds):
    if preds is None:
        return
        
    # Handle different prediction formats
    if len(preds.shape) > 1:
        preds = preds[0]
    
    # Check if we have 5 classes (including the extra bolt class)
    if len(preds) == 5:
        bolt_score = preds[0] + preds[4]
        bearing_score = preds[1]
        nut_score = preds[2]
        gear_score = preds[3]
        merged_preds = [bolt_score, bearing_score, nut_score, gear_score]
    else:
        # Handle 4 classes directly
        merged_preds = list(preds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, merged_preds, color='teal', edgecolor='black')
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("🔍 Prediction Probabilities", fontsize=14)
    for i, value in enumerate(merged_preds):
        ax.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=10)
    st.pyplot(fig)
    plt.close(fig) 
# --------------------- Streamlit UI ---------------------
st.set_page_config("🛠️ Mechanical Part Identifier", layout="centered")
st.title("🛠️ Mechanical Part Identifier")
st.markdown("<p style='text-align:center;'>Upload or capture an image to identify the mechanical part (Bolt, Bearing, Nut, Gear).</p>", unsafe_allow_html=True)
st.markdown("---")

model = load_model()

if model is None:
    st.error("❌ Failed to load the model. Please check if 'model.savedmodel' exists.")
    st.stop()

# --------------------- Camera or Upload Input ---------------------
image = None
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)

with col2:
    camera_image = st.camera_input("📷 Or take a photo")
if camera_image and not image:
    image = Image.open(camera_image)

# --------------------- Prediction ---------------------
if image:
    st.markdown("### 📸 Image Preview")
    st.image(image, width=300, caption="Preview", use_container_width=False)

    with st.spinner("🔍 Analyzing image..."):
        input_tensor = preprocess_image(image)
        predictions = make_prediction(model, input_tensor)

    if predictions is not None:
        # Handle different prediction formats
        if len(predictions.shape) > 1:
            preds = predictions[0]
        else:
            preds = predictions

        # Merge predictions if we have 5 classes
        if len(preds) == 5:
            bolt_score = preds[0] + preds[4]
            bearing_score = preds[1]
            nut_score = preds[2]
            gear_score = preds[3]
            merged_logits = np.array([bolt_score, bearing_score, nut_score, gear_score])
        else:
            merged_logits = preds

        predicted_index = np.argmax(merged_logits)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(merged_logits[predicted_index])

        st.markdown("---")
        st.success(f"✅ Predicted Part: **{predicted_class}**")
        st.info(f"🔢 Confidence: **{confidence * 100:.2f}%**")

        show_top_predictions(predictions)
        plot_predictions(predictions)
        show_part_info(predicted_class)

        # Store prediction history in session
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "class": predicted_class,
            "confidence": f"{confidence*100:.2f}%"
        })
    else:
        st.error("❌ Failed to make prediction. Please try again.")

# --------------------- Prediction History ---------------------
if "history" in st.session_state and st.session_state.history:
    st.markdown("### 🕘 Prediction History")
    st.table(st.session_state.history)

# --------------------- Sample Images ---------------------
st.markdown("---")
st.markdown("### 🧪 Try with Sample Images")
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
                    st.image(filepath, caption=label, width=150, use_container_width=False)
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