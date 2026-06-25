import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import io

from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
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
# PDF REPORT
# ---------------------------------

def create_pdf(result, confidence):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "<b>PNEUMOVISION AI</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "Clinical Decision Support Report",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Prediction:</b> {result}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Clinical Interpretation</b>",
            styles["Heading2"]
        )
    )

    if result == "PNEUMONIA":

        story.append(
            Paragraph(
                "The uploaded chest X-ray demonstrates imaging patterns associated with pneumonia. Clinical evaluation is recommended.",
                styles["BodyText"]
            )
        )

    else:

        story.append(
            Paragraph(
                "No significant radiographic evidence of pneumonia was identified by the model.",
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph(
            "<b>Model Information</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "Architecture: MobileNetV2",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "Input Size: 160 × 160",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "Overall Accuracy: 80%",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<br/><br/>Disclaimer: This report is generated for research and educational purposes only and should not replace professional medical diagnosis.",
            styles["Italic"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer
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
# PROFESSIONAL SIDEBAR
# ---------------------------------

st.sidebar.title("🩺 PneumoVision AI")

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 AI Model")

st.sidebar.success("""
**Architecture**

MobileNetV2
""")

st.sidebar.info("""
**Transfer Learning**

ImageNet Pretrained

**Version**

1.0

**Framework**

TensorFlow • Keras
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Performance")

c1, c2 = st.sidebar.columns(2)

c1.metric("Accuracy", "80%")
c2.metric("Precision", "84%")

c3, c4 = st.sidebar.columns(2)

c3.metric("Recall", "80%")
c4.metric("F1", "78%")

st.sidebar.markdown("---")

st.sidebar.subheader("🖼 Input Specification")

st.sidebar.write("Resolution : **160 × 160**")

st.sidebar.write("Channels : **RGB**")

st.sidebar.write("Classes :")

st.sidebar.success("🟢 Normal")

st.sidebar.error("🔴 Pneumonia")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙ AI Workflow")

st.sidebar.markdown("""
📤 Upload X-ray

⬇

🖼 Resize Image

⬇

🎨 Normalize Pixels

⬇

🧠 MobileNetV2

⬇

📊 Prediction

⬇

📄 Clinical Report
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Clinical Decision Support System\nVersion 1.0"
)

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

    score = float(prediction[0][0])

    with col2:

        st.markdown("""
# 🧠 AI Diagnostic Report
""")

        if score > 0.5:

            confidence = score * 100
            result = "PNEUMONIA"

            st.markdown("""
            ### 🫁 Prediction

            # **PNEUMONIA**
            """)

            st.error("Positive Screening Result")

        else:

            confidence = (1 - score) * 100
            result = "NORMAL"

            st.markdown("""
            ### ✅ Prediction

            # **NORMAL**
            """)

            st.success("No Significant Abnormality Detected")

        st.markdown("### 🎯 Confidence Score")

        st.markdown(
            f"<h1 style='color:#1f77b4'>{confidence:.2f}%</h1>",
            unsafe_allow_html=True
            )

        st.progress(confidence / 100)
        st.markdown("### 🚦 Clinical Risk Level")
        if result == "PNEUMONIA":

            if confidence >= 90:

                st.error("🔴 HIGH RISK")

            elif confidence >= 75:

                st.warning("🟠 MODERATE RISK")

            else:

                st.info("🟡 LOW RISK")

    else:

        st.success("🟢 NORMAL")
        
        
        st.markdown("## 📊 AI Probability Analysis")

        if result == "PNEUMONIA":

            left, right = st.columns(2)

            with left:
                st.success("🫁 Pneumonia")
                st.markdown(
            f"<h2 style='color:#d62728;'>{confidence:.2f}%</h2>",
            unsafe_allow_html=True
        )

            with right:
                st.info("✅ Normal")
                st.markdown(
            f"<h2 style='color:#2ca02c;'>{100-confidence:.2f}%</h2>",
            unsafe_allow_html=True
        )

    else:

        left, right = st.columns(2)

        with left:
            st.success("✅ Normal")
            st.markdown(
            f"<h2 style='color:#2ca02c;'>{confidence:.2f}%</h2>",
            unsafe_allow_html=True
        )

        with right:
            st.info("🫁 Pneumonia")
            st.markdown(
            f"<h2 style='color:#d62728;'>{100-confidence:.2f}%</h2>",
            unsafe_allow_html=True
        )

        # ---------------------------------
        # DASHBOARD METRICS
        # ---------------------------------


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
    st.markdown("---")

    pdf = create_pdf(
    result,
    confidence
)

    st.download_button(

        "📄 Download Clinical Report",

        pdf,

        file_name="PneumoVision_Report.pdf",

        mime="application/pdf"
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
