import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import cv2
import os
from matplotlib.image import imread
from scipy import ndimage as nd
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide")

# Load YOLO model=
yolo_model = YOLO("runs/segment/train/weights/best.pt") 

# Load classification model
model_path = os.path.join(os.path.dirname(__file__), 'model.keras')
model = tf.keras.models.load_model(model_path)

# Preprocess uploaded image
def preprocess_image(uploaded_image):
    img = load_img(uploaded_image, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize

# Classification function
def classify_image(img_array):
    return model.predict(img_array)

# Segment image using HSV and YOLO
def segment_image(uploaded_image):
    input_image = imread(uploaded_image)
    input_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # ==== HSV-based segmentation ====
    hsv = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2HSV)
    damage_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 65))
    closed_damage_mask = nd.binary_closing(damage_mask, np.ones((12, 12)))

    chip_mask = cv2.inRange(hsv, (20, 50, 110), (30, 255, 255))
    closed_chip_mask = nd.binary_closing(chip_mask, np.ones((3, 3)))

    red_mask = np.zeros_like(input_bgr)
    red_mask[closed_damage_mask == 1] = [0, 0, 255]
    hsv_overlay = cv2.addWeighted(input_bgr, 1.0, red_mask, 0.4, 0)

    damaged_pixels_hsv = np.sum(closed_damage_mask == 1)
    undamaged_pixels_hsv = np.sum(closed_chip_mask == 1)
    total_pixels_hsv = undamaged_pixels_hsv + damaged_pixels_hsv
    damage_percentage_hsv = damaged_pixels_hsv / total_pixels_hsv if total_pixels_hsv > 0 else 0

    # ==== YOLO-based segmentation ====
    yolo_results = yolo_model.predict(input_bgr, imgsz=640, verbose=False)
    yolo_overlay = input_bgr.copy()

    chip_pixels_yolo = 0
    damaged_pixels_yolo = 0

    for r in yolo_results:
        if r.masks is not None:
            for mask, cls_id in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                cls_name = yolo_model.names[int(cls_id)]
                mask_resized = cv2.resize(mask, (input_bgr.shape[1], input_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = (mask_resized > 0.5).astype(np.uint8)

                # Count chip and damage area
                if cls_name == "chip":
                    chip_pixels_yolo += np.sum(mask_resized == 1)
                    continue

                elif cls_name in ["damaged", "semi-damaged"]:
                    damaged_pixels_yolo += np.sum(mask_resized == 1)

                # Overlay colors (only for damaged/semi-damaged)
                color = (0, 0, 255) if cls_name == "damaged" else (0, 165, 255)
                color_mask = np.zeros_like(input_bgr)
                color_mask[mask_resized == 1] = color
                yolo_overlay = cv2.addWeighted(yolo_overlay, 1, color_mask, 0.4, 0)

    damage_percentage_yolo = damaged_pixels_yolo / chip_pixels_yolo if chip_pixels_yolo > 0 else 0

    return damage_percentage_hsv, hsv_overlay, damage_percentage_yolo, yolo_overlay

# ==== Streamlit UI ====
st.title("Potato Chip Damage Detection")

st.markdown(
    """
    <style>
    /* Set the background image */
    .stApp {
        background-image: url('https://img.freepik.com/free-photo/flat-lay-beer-bottles-with-chips-nuts_23-2148754981.jpg?t=st=1737990572~exp=1737994172~hmac=e8b153437b6ef164d138d6caac73be5d1776fa91bd4d9e005e50eb6115f3bc3b&w=1060');
        background-size: cover;
        background-position: center;
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    }
    </style>
    """,
    unsafe_allow_html=True
)

test_images = {
    "Sample Defective Chip 1": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_004846.jpg'),
    "Sample Defective Chip 2": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_010328.jpg'),
    "Sample Defective Chip 3": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Defective/IMG_20210319_004823.jpg'),
    "Sample Not Defective Chip 1": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_231229.jpg'),
    "Sample Not Defective Chip 2": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_231650.jpg'),
    "Sample Not Defective Chip 3": os.path.join(os.path.dirname(__file__), 'Pepsico/Test/Not Defective/IMG_20210318_232125.jpg')
}

with st.sidebar:
    st.write("### Upload your potato chip image")
    uploaded_image = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'webp'])

    st.write("### Or Select a Sample Image:")
    selected_test_image = None
    for label, path in test_images.items():
        if st.button(f"🔍 {label}"):
            selected_test_image = path

image_to_process = uploaded_image if uploaded_image else selected_test_image

if image_to_process:
    try:
        
        img_array = preprocess_image(image_to_process)
        prediction = classify_image(img_array)
        damage_percentage_hsv, hsv_overlay, damage_percentage_yolo, yolo_overlay = segment_image(image_to_process)
        
        col1, col2, col3 = st.columns([1.5, 1.5, 1.5], gap="medium")

        with col1:
            image = Image.open(image_to_process) if hasattr(image_to_process, 'read') else Image.open(str(image_to_process))
            st.image(image, use_container_width=True)
            st.markdown("<p style='text-align:center; color:white; font-size:18px;'>Original Image</p>", unsafe_allow_html=True)

        with col2:
            st.image(cv2.cvtColor(hsv_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<p style='text-align:center; color:white; font-size:18px;'>HSV Damage ({damage_percentage_hsv:.2%})</p>", unsafe_allow_html=True)

        with col3:
            st.image(cv2.cvtColor(yolo_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<p style='text-align:center; color:white; font-size:18px;'>YOLO Damage ({damage_percentage_yolo:.2%})</p>", unsafe_allow_html=True)


        # Display classification result
        if prediction[0] > 0.5:
            st.markdown(
                "<h2 style='color:#75F94D; text-align:center;'>This chip is not damaged.</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:#ED0025; text-align:center;'>This chip is damaged.</h2>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
