import streamlit as st
import torch
import shutil
import glob
import os
from PIL import Image

# ---------------------------
# Page Config and Title
# ---------------------------
st.set_page_config(page_title="Crop & Weed Detection", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: green;'>🌾 Crop & Weed Detection</h1>
    <p style='text-align: center;'>Upload an image to identify whether it's a healthy crop or has weeds.</p>
""", unsafe_allow_html=True)

# ---------------------------
# Load YOLOv5 Model
# ---------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='results/cropweed_test2/weights/best.pt', force_reload=True)

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload a field image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with open("input.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    st.image("input.jpg", caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Clear previous results
    # ---------------------------
    if os.path.exists("runs/detect/exp"):
        shutil.rmtree("runs/detect/exp")

    # ---------------------------
    # Perform detection
    # ---------------------------
    with st.spinner("Detecting..."):
        results = model("input.jpg")
        results.save()

    # ---------------------------
    # Get Result Image
    # ---------------------------
    detected_image = glob.glob("runs/detect/exp/*.jpg")[0]
    st.image(detected_image, caption="🧪 Detection Output", use_container_width=True)

    # ---------------------------
    # Parse results
    # ---------------------------
    labels = results.pandas().xyxy[0]['name'].tolist()

    if len(labels) == 0:
        st.success("✅ **No weed detected! The crop is healthy.**")
    elif "weed" in [l.lower() for l in labels]:
        st.error("❌ **Weed detected! Immediate action required.**")
    else:
        st.warning(f"⚠️ Detected: {', '.join(labels)}")

    # ---------------------------
    # Show raw predictions (Optional)
    # ---------------------------
    with st.expander("🔍 View Detection Details"):
        st.dataframe(results.pandas().xyxy[0][['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
