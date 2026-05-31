import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image

from huggingface_hub import hf_hub_download

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input


st.set_page_config(
    page_title="Brain Tumor Trustworthy AI",
    layout="wide"
)

st.title(
    "🧠 Brain Tumor Classification using Trustworthy AI"
)

st.write(
    "Prediction + Confidence + Uncertainty + Grad-CAM"
)
