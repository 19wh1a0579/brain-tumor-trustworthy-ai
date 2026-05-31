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



@st.cache_resource
def load_brain_tumor_model():

    model_path = hf_hub_download(
        repo_id="suvidha-reddy/brain-tumor-trustworthy-ai",
        filename="brain_tumor_resnet152v2.h5"
    )

    model = load_model(model_path)

    return model

loaded_model = load_brain_tumor_model()

st.success("Model Loaded Successfully")



class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]



resnet_model = loaded_model.get_layer("resnet152v2")

grad_backbone = tf.keras.Model(
    inputs=resnet_model.input,
    outputs=resnet_model.get_layer("post_relu").output
)




