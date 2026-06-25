import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import matplotlib.cm as cm
# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

MODEL_PATH = "mobilenetv2_original.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
# ---------------------------------
# GRAD CAM
# ---------------------------------

def make_gradcam_heatmap(img_array, model):

    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer("out_relu")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ---------------------------------
# HEADER
# ---------------------------------

st.title("🩺 PneumoVision AI")
st.subheader("AI-Powered Clinical Decision Support System")

st.write(
    """
    This system analyzes chest X-ray images using a MobileNetV2-based
    deep learning model and provides automated pneumonia screening
    predictions with confidence scores.
    """
)

st.markdown("---")

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.header("Model Information")

st.sidebar.success("Model: MobileNetV2")

st.sidebar.info("""
Input Size: 160 × 160

Classes:
• Normal
• Pneumonia

Framework:
• TensorFlow / Keras
""")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

st.info(
    "Upload a chest X-ray image (JPG, JPEG, or PNG) to generate a pneumonia screening prediction."
)

# ---------------------------------
# PREDICTION PIPELINE
# ---------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Chest X-Ray",
            width=450
        )

        width, height = image.size

        st.markdown("### Image Information")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Width", width)

        c2.metric("Height", height)

        c3.metric("Format", image.format)

        file_size = round(uploaded_file.size / 1024, 2)
        c4.metric("Size (KB)", file_size)

    # ---------------------------------
    # PREPROCESS
    # ---------------------------------

    img = np.array(image)

    img = cv2.resize(
        img,
        (160, 160)
    )

    img = img.astype("float32") / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # ---------------------------------
    # PREDICTION
    # ---------------------------------

    prediction = model.predict(img, verbose=0)
    
    heatmap = make_gradcam_heatmap(img, model)

    score = float(prediction[0][0])

    with col2:

        st.subheader("Prediction Result")

        if score > 0.5:

            confidence = score * 100
            result = "PNEUMONIA"

            st.error(
                "Prediction: PNEUMONIA"
            )

        else:

            confidence = (1 - score) * 100
            result = "NORMAL"

            st.success(
                "Prediction: NORMAL"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)
        st.markdown("### Diagnostic Assessment")

        if result == "PNEUMONIA":

            st.write(
        f"🫁 Pneumonia Probability: {confidence:.2f}%"
    )

            st.write(
        f"✅ Normal Probability: {100-confidence:.2f}%"
    )

        else:

            st.write(
        f"✅ Normal Probability: {confidence:.2f}%"
    )

            st.write(
        f"🫁 Pneumonia Probability: {100-confidence:.2f}%"
    )

        # ---------------------------------
        # DASHBOARD METRICS
        # ---------------------------------

        st.markdown("### Model Summary")
        st.caption(
    "Image resized to 160×160 and normalized before inference."
)
        m1, m2, m3 = st.columns(3)

        m1.metric("Model", "MobileNetV2")
        m2.metric("Classes", "2")
        m3.metric("Input", "160×160")
        
        st.markdown("### Model Performance")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("Accuracy", "80%")
        p2.metric("Precision", "84%")
        p3.metric("Recall", "80%")
        p4.metric("F1 Score", "78%")

        # ---------------------------------
        # CONFIDENCE LEVEL
        # ---------------------------------

        if result == "PNEUMONIA":

            if confidence >= 90:
                st.error("High Risk Assessment")

            elif confidence >= 75:
                st.warning("Moderate Risk Assessment")

            else:
                st.info("Low Risk Assessment")

        else:

            st.success("Normal Screening Result")

    # ---------------------------------
    # CLINICAL INTERPRETATION
    # ---------------------------------
    st.markdown("---")

    st.subheader("Explainable AI (Grad-CAM)")

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(
        np.uint8(jet_heatmap * 255)
)

    jet_heatmap = jet_heatmap.resize(image.size)

    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = (
        jet_heatmap * 0.4 +
        np.array(image)
)

    superimposed_img = np.uint8(superimposed_img)

    c1, c2 = st.columns(2)

    with c1:

        st.image(
        image,
        caption="Original Chest X-Ray"
    )

    with c2:

        st.image(
        superimposed_img,
        caption="AI Attention Heatmap"
    )

    st.info(
    "Highlighted regions indicate the areas that contributed most to the model's prediction."
)
    
    st.markdown("---")
    st.subheader("Screening Status")

    if result == "PNEUMONIA":

        st.error(
        "Potential Pneumonia Detected"
    )

    else:

        st.success(
        "No Significant Pneumonia Indicators"
    )
    st.subheader("Clinical Interpretation")

    if result == "PNEUMONIA":

        st.warning(
            """
            The model detected radiographic patterns commonly associated
            with pneumonia. Further clinical evaluation and expert review
            are recommended.
            """
        )

    else:

        st.success(
            """
            No strong radiographic evidence of pneumonia was identified
            by the model. Clinical correlation is recommended.
            """
        )

    # ---------------------------------
    # RECOMMENDATION
    # ---------------------------------

    st.subheader("Clinical Recommendation")

    if result == "PNEUMONIA":

        st.error(
            """
            Recommend further diagnostic assessment by a qualified
            healthcare professional.
            """
        )

    else:

        st.info(
            """
            No significant signs of pneumonia detected by the model.
            """
        )

# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")

st.caption(
    f"Report Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)
st.caption(
    """
    Disclaimer:
    This Clinical Decision Support System is intended for educational
    and research purposes only. Predictions should not replace
    professional medical diagnosis.
    """
)
