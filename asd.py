
import streamlit as st
import numpy as np
import pickle
from PIL import Image

@st.cache_resource
def load_model():
    model_path = "/home/sahildogra/Downloads/train1.py/saved_models/log_reg_20250820_133401.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# find the expected input features from model
n_features = model.n_features_in_  # e.g., 12288

# infer image size
img_size = int(np.sqrt(n_features)) if int(np.sqrt(n_features))**2 == n_features else None
if img_size is None:
    # probably RGB flatten
    img_size = int(np.sqrt(n_features // 3))
    channels = 3
else:
    channels = 1

st.title("🩻 Pneumonia Detection App (Local Model)")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if channels == 1:
        image = image.convert("L")  # grayscale
    else:
        image = image.convert("RGB")  # RGB

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to match training size
    image = image.resize((img_size, img_size))  
    img_array = np.array(image) / 255.0

    if channels == 1:
        img_array = img_array.reshape(1, -1)  # flatten grayscale
    else:
        img_array = img_array.reshape(1, -1)  # flatten RGB

    prediction = model.predict(img_array)

    if prediction[0] == 0:
        st.success("✅ Prediction: NORMAL")
    else:
        st.error("⚠️ Prediction: PNEUMONIA")
