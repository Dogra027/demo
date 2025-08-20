import streamlit as st
import pickle
import cv2
import numpy as np

# ✅ Load model from GitHub repo (keep the .pkl file in same folder as this file)
@st.cache_resource
def load_model():
    model_path = "log_reg_20250820_133401.pkl"  # relative path
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🖼️ Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Preprocess image (⚠️ adjust according to how you trained your model)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))   # example resize
    flattened = resized.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(flattened)[0]
    st.success(f"✅ Prediction: {prediction}")
