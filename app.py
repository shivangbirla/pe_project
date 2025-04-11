import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configure page
st.set_page_config(page_title="AI Image Processor", layout="wide", page_icon="🎨")
st.title("🎨 AI-Powered Image Processing")

# Custom CSS for modern UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(45deg, #1a1a1a, #2a2a2a) !important;
        color: white !important;
    }
    .stButton>button {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# Image processing functions
def negative_image(img):
    return 255 - img

def custom_convolution(image, kernel):
    # Check if image is color (3D)
    if image.ndim == 3:
        # Apply the convolution channel-wise
        return np.stack([custom_convolution(image[:, :, c], kernel) for c in range(image.shape[2])], axis=2)

    kernel_height, kernel_width = kernel.shape
    pad = kernel_height // 2

    # Pad the image (grayscale)
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = image_padded[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output


def average_smoothing_5x5(img):
    kernel = np.ones((5, 5), np.float32) / 25
    return custom_convolution(img, kernel)

def median_smoothing(img, size=3):
    return cv2.medianBlur(img, size)

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

def gaussian_smoothing(img, sigma=1.0):
    return cv2.GaussianBlur(img, (5, 5), sigma)

def edge_detection(img):
    return cv2.Canny(img, 100, 200)

def resize_image(img, scale_factor):
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

# Sidebar UI
with st.sidebar:
    st.header("📤 Upload Your Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        st.header("🎚 Processing Controls")
        processor = st.radio("Select Operation:", [
            "Negative Image", "Average Smoothing (5x5)", "Median Smoothing",
            "Rotate Image", "Gaussian Smoothing", "Edge Detection", "Resize Image"
        ])
        
        if processor in ["Rotate Image"]:
            angle = st.slider("Rotation Angle", -180, 180, 0, 5)
        elif processor in ["Gaussian Smoothing"]:
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        elif processor in ["Resize Image"]:
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)

if uploaded_file:
    image = np.array(Image.open(uploaded_file))  
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("🖼 Original Image")
        st.image(image, use_column_width=True, clamp=True)
    
    processed = image.copy()
    if processor == "Negative Image":
        processed = negative_image(image)
    elif processor == "Average Smoothing (5x5)":
        processed = average_smoothing_5x5(image)
    elif processor == "Median Smoothing":
        processed = median_smoothing(image, 3)
    elif processor == "Rotate Image":
        processed = rotate_image(image, angle)
    elif processor == "Gaussian Smoothing":
        processed = gaussian_smoothing(image, sigma)
    elif processor == "Edge Detection":
        processed = edge_detection(image)
    elif processor == "Resize Image":
        processed = resize_image(image, scale_factor)
    
    with col2:
        st.subheader("🎨 Processed Image")
        st.image(processed, use_column_width=True, clamp=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px">
        <h2 style="color: #666">📁 Upload an Image to Start</h2>
        <p style="color: #444">Supports JPG, PNG, JPEG formats</p>
    </div>
    """, unsafe_allow_html=True)