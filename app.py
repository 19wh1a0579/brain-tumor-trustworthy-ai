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


@st.cache_resource
def load_brain_tumor_model():

    model_path = hf_hub_download(
        repo_id="suvidha-reddy/brain-tumor-trustworthy-ai",
        filename="brain_tumor_resnet152v2.h5",
        repo_type="model"
    )

    model = load_model(model_path)

    return model

loaded_model = load_brain_tumor_model()




class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]
tumor_names = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor"
}



resnet_model = loaded_model.get_layer("resnet152v2")

grad_backbone = tf.keras.Model(
    inputs=resnet_model.input,
    outputs=resnet_model.get_layer("post_relu").output
)



bn_layer = loaded_model.get_layer(
    "batch_normalization"
)

dense_layer = loaded_model.get_layer(
    "dense"
)

output_layer = loaded_model.get_layer(
    "dense_1"
)

feature_input = tf.keras.Input(
    shape=(7,7,2048)
)

x = tf.keras.layers.GlobalAveragePooling2D()(
    feature_input
)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

predictions = tf.keras.layers.Dense(
    4,
    activation="softmax"
)(x)

classifier_model = tf.keras.Model(
    feature_input,
    predictions
)

classifier_model.layers[2].set_weights(
    bn_layer.get_weights()
)

classifier_model.layers[3].set_weights(
    dense_layer.get_weights()
)

classifier_model.layers[4].set_weights(
    output_layer.get_weights()
)

st.sidebar.success("Grad-CAM Components Ready")


def generate_gradcam(feature_maps,
                     classifier_model):

    feature_maps_tensor = tf.convert_to_tensor(
        feature_maps,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(feature_maps_tensor)

        preds = classifier_model(
            feature_maps_tensor,
            training=False
        )

        pred_index = tf.argmax(
            preds[0]
        )

        class_score = preds[:, pred_index]

    grads = tape.gradient(
        class_score,
        feature_maps_tensor
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    feature_map = feature_maps_tensor[0]

    heatmap = tf.reduce_sum(
        feature_map * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def create_overlay(original_img,
                   heatmap):

    heatmap_resized = cv2.resize(
        heatmap,
        (
            original_img.shape[1],
            original_img.shape[0]
        )
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_img,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


def predict_and_explain(img):

    img_resized = img.resize(
        (224,224)
    )

    img_array = image.img_to_array(
        img_resized
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # ------------------
    # Prediction
    # ------------------

    probs = loaded_model.predict(
        img_array,
        verbose=0
    )[0]

    pred_idx = np.argmax(
        probs
    )

    pred_class = class_names[
        pred_idx
    ]

    confidence = float(
        np.max(probs) * 100
    )

    # ------------------
    # Entropy
    # ------------------

    probs_safe = np.clip(
        probs,
        1e-10,
        1.0
    )

    entropy = float(
        -np.sum(
            probs_safe *
            np.log(probs_safe)
        )
    )

    if entropy < 0.5:
        uncertainty = "LOW"

    elif entropy < 1.0:
        uncertainty = "MEDIUM"

    else:
        uncertainty = "HIGH"

    # ------------------
    # Grad-CAM
    # ------------------

    feature_maps = grad_backbone.predict(
        img_array,
        verbose=0
    )

    heatmap = generate_gradcam(
        feature_maps,
        classifier_model
    )

    return (
        pred_class,
        confidence,
        entropy,
        uncertainty,
        heatmap
    )


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

tumor_names = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor"
}

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    original_img = np.array(img)

    with st.spinner("Analyzing MRI..."):

        pred_class, confidence, entropy, uncertainty, heatmap = predict_and_explain(img)

        overlay = create_overlay(
            original_img,
            heatmap
        )

    st.subheader("MRI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            original_img,
            caption="Original MRI Scan",
            width=350
        )

    with col2:
        st.image(
            overlay,
            caption="Grad-CAM Explainability",
            width=350
        )

    st.subheader("Prediction Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Prediction",
            tumor_names[pred_class]
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with c3:
        st.metric(
            "Entropy",
            f"{entropy:.4f}"
        )

    with c4:
        st.metric(
            "Uncertainty",
            uncertainty
        )

    st.markdown("---")

    if confidence > 95:
        st.success(
            "✅ Very High Confidence Prediction"
        )

    elif confidence > 80:
        st.info(
            "ℹ️ High Confidence Prediction"
        )

    else:
        st.warning(
            "⚠️ Low Confidence Prediction"
        )

    if uncertainty == "LOW":
        st.success(
            "🟢 Uncertainty Level: LOW"
        )

    elif uncertainty == "MEDIUM":
        st.warning(
            "🟡 Uncertainty Level: MEDIUM"
        )

    else:
        st.error(
            "🔴 Uncertainty Level: HIGH"
        )
