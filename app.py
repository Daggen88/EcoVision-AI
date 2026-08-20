import streamlit as st
import cv2
import numpy as np
import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide"
)


# ============================================================
# ROBOFLOW CONFIG
# ============================================================

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

# Streamlit Cloud Secrets
if not api_key:
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "❌ ROBOFLOW_API_KEY belum dikonfigurasi.\n\n"
        "Masukkan API key di Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)


# ============================================================
# ROBOFLOW WORKFLOW
# ============================================================

WORKSPACE_NAME = "daggen580-gmail-com"

WORKFLOW_ID = (
    "my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic"
)


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(image):
    """
    Menjalankan Roboflow Workflow.
    """

    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={
            "image": image
        }
    )

    return result


# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

def find_predictions(data):
    """
    Mencari predictions dari response Roboflow Workflow.

    Fungsi ini sengaja dibuat recursive supaya aman terhadap
    response seperti:

    {
        "outputs": [
            {
                "predictions": {
                    "predictions": [...]
                }
            }
        ]
    }

    atau bentuk nested lainnya.
    """

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if isinstance(data, list):

        # Cek apakah list ini langsung berisi predictions
        for item in data:

            if isinstance(item, dict):

                if (
                    "class" in item
                    and (
                        "confidence" in item
                        or "score" in item
                    )
                ):
                    return data

        # Kalau belum ketemu, cari lebih dalam
        for item in data:

            result = find_predictions(item)

            if result:
                return result

        return []


    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(data, dict):

        # Kalau dictionary ini sendiri adalah prediction
        if (
            "class" in data
            and (
                "confidence" in data
                or "score" in data
            )
        ):

            return [data]


        # Cari predictions dulu
        if "predictions" in data:

            result = find_predictions(
                data["predictions"]
            )

            if result:
                return result


        # Cari outputs
        if "outputs" in data:

            result = find_predictions(
                data["outputs"]
            )

            if result:
                return result


        # Cari seluruh value dictionary
        for value in data.values():

            if isinstance(
                value,
                (dict, list)
            ):

                result = find_predictions(
                    value
                )

                if result:
                    return result

        return []


    return []


# ============================================================
# CATEGORY
# ============================================================

def get_category(label):

    label = str(label).lower().strip()

    if label == "battery":

        return (
            "B3",
            "Buang ke tempat khusus B3"
        )

    elif label in [
        "cardboard",
        "paper",
        "plastic",
        "metal",
        "glass"
    ]:

        return (
            "ANORGANIK",
            "Buang ke tempat sampah ANORGANIK"
        )

    elif label in [
        "organic",
        "food",
        "food-waste",
        "organic-waste"
    ]:

        return (
            "ORGANIK",
            "Buang ke tempat sampah ORGANIK"
        )

    elif label == "trash":

        return (
            "GENERAL WASTE",
            "Buang ke tempat sampah umum"
        )

    else:

        return (
            "UNKNOWN",
            "Belum dikenali"
        )


# ============================================================
# DRAW PREDICTIONS
# ============================================================

def draw_predictions(image, predictions):

    annotated = image.copy()

    if not predictions:
        return annotated

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict
        ):
            continue


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = prediction.get(
            "class",
            prediction.get(
                "label",
                "unknown"
            )
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = prediction.get(
            "confidence",
            prediction.get(
                "score",
                0
            )
        )

        try:

            confidence = float(
                confidence
            )

        except Exception:

            confidence = 0.0


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x = prediction.get("x")
        y = prediction.get("y")

        width = prediction.get(
            "width"
        )

        height = prediction.get(
            "height"
        )


        if any(
            value is None
            for value in [
                x,
                y,
                width,
                height
            ]
        ):

            continue


        try:

            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)

        except Exception:

            continue


        # ----------------------------------------------------
        # CENTER → CORNER
        # ----------------------------------------------------

        x1 = int(
            x - width / 2
        )

        y1 = int(
            y - height / 2
        )

        x2 = int(
            x + width / 2
        )

        y2 = int(
            y + height / 2
        )


        # ----------------------------------------------------
        # KEEP INSIDE IMAGE
        # ----------------------------------------------------

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            image.shape[1] - 1,
            x2
        )

        y2 = min(
            image.shape[0] - 1,
            y2
        )


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # LABEL TEXT
        # ----------------------------------------------------

        text = (
            f"{str(label).upper()} "
            f"{confidence * 100:.1f}%"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.8

        thickness = 2


        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )


        label_y1 = max(
            0,
            y1 - text_height - baseline - 8
        )

        label_y2 = y1

        label_x2 = min(
            image.shape[1],
            x1 + text_width + 12
        )


        # ----------------------------------------------------
        # LABEL BACKGROUND
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (label_x2, label_y2),
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # LABEL TEXT
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            text,
            (
                x1 + 6,
                y1 - 8
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )


    return annotated


# ============================================================
# DETECTION INFORMATION
# ============================================================

def show_detection_metrics(
    predictions
):

    if not predictions:

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Detected Object",
                "No Object"
            )

        with col2:

            st.metric(
                "Confidence",
                "0.0%"
            )

        with col3:

            st.metric(
                "Category",
                "-"
            )

        with col4:

            st.metric(
                "Objects",
                "0"
            )

        st.warning(
            "⚠️ Tidak ada objek yang terdeteksi."
        )

        return


    # --------------------------------------------------------
    # VALID PREDICTIONS
    # --------------------------------------------------------

    valid_predictions = []

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict
        ):
            continue


        label = prediction.get(
            "class",
            prediction.get(
                "label"
            )
        )

        confidence = prediction.get(
            "confidence",
            prediction.get(
                "score",
                0
            )
        )


        if label is None:
            continue


        try:

            confidence = float(
                confidence
            )

        except Exception:

            confidence = 0.0


        valid_predictions.append(
            {
                "label": str(label),
                "confidence": confidence
            }
        )


    if not valid_predictions:

        st.warning(
            "⚠️ Tidak ada objek yang terdeteksi."
        )

        return


    # --------------------------------------------------------
    # BEST DETECTION
    # --------------------------------------------------------

    best_prediction = max(
        valid_predictions,
        key=lambda item:
        item["confidence"]
    )


    label = best_prediction[
        "label"
    ]

    confidence = best_prediction[
        "confidence"
    ]


    category, recommendation = (
        get_category(label)
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Detected Object",
            label.upper()
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%"
        )


    with col3:

        st.metric(
            "Category",
            category
        )


    with col4:

        st.metric(
            "Objects",
            len(valid_predictions)
        )


    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    st.info(
        f"♻️ Rekomendasi: {recommendation}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "♻️ EcoVision AI"
)

st.sidebar.markdown(
    "### Navigation"
)

menu = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📷 Live Detection",
        "🖼️ Image Detection",
        "ℹ️ About"
    ]
)


# ============================================================
# HOME
# ============================================================

if menu == "🏠 Home":

    st.title(
        "♻️ EcoVision AI"
    )

    st.subheader(
        "AI-Powered Waste Detection"
    )

    st.write(
        """
        EcoVision AI adalah aplikasi Computer Vision
        untuk mendeteksi jenis sampah secara otomatis
        menggunakan AI.
        """
    )

    st.markdown(
        """
        ### Fitur

        📷 **Live Detection**  
        Gunakan kamera untuk mendeteksi sampah.

        🖼️ **Image Detection**  
        Upload gambar sampah.

        ♻️ **Smart Classification**  
        Sistem menentukan kategori sampah dan
        memberikan rekomendasi pembuangan.
        """
    )


# ============================================================
# LIVE DETECTION
# ============================================================

elif menu == "📷 Live Detection":

    st.title(
        "📷 Live Detection"
    )

    st.write(
        "Gunakan kamera perangkat untuk mengambil "
        "gambar dan mendeteksi jenis sampah."
    )


    # --------------------------------------------------------
    # HIDE CAMERA AFTER CAPTURE
    # --------------------------------------------------------

    camera_image = st.camera_input(
        "Ambil gambar menggunakan kamera"
    )


    if camera_image is not None:

        # Hide captured camera widget so that only
        # the annotated result remains visible.

        st.markdown(
            """
            <style>
            div[data-testid="stCameraInput"] {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # RETAKE BUTTON
        # ----------------------------------------------------

        if st.button(
            "📷 Ambil Foto Baru"
        ):

            st.rerun()


        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                camera_image.getvalue()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "❌ Gambar tidak dapat dibaca."
            )

        else:

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    result = run_inference(
                        image
                    )

                    predictions = (
                        find_predictions(
                            result
                        )
                    )

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []


            # ------------------------------------------------
            # ANNOTATE IMAGE
            # ------------------------------------------------

            annotated_image = (
                draw_predictions(
                    image,
                    predictions
                )
            )


            # ------------------------------------------------
            # ONE IMAGE ONLY
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Result"
            )

            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            # ------------------------------------------------
            # INFORMATION
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Information"
            )

            show_detection_metrics(
                predictions
            )


# ============================================================
# IMAGE DETECTION
# ============================================================

elif menu == "🖼️ Image Detection":

    st.title(
        "🖼️ Image Detection"
    )

    uploaded_file = st.file_uploader(
        "Upload gambar sampah",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        file_bytes = np.asarray(
            bytearray(
                uploaded_file.read()
            ),
            dtype=np.uint8
        )


        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "❌ Gambar tidak dapat dibaca."
            )

        else:

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    result = run_inference(
                        image
                    )

                    predictions = (
                        find_predictions(
                            result
                        )
                    )

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []


            # ------------------------------------------------
            # ANNOTATED IMAGE
            # ------------------------------------------------

            annotated_image = (
                draw_predictions(
                    image,
                    predictions
                )
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Result"
            )

            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            # ------------------------------------------------
            # INFORMATION
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Information"
            )

            show_detection_metrics(
                predictions
            )


# ============================================================
# ABOUT
# ============================================================

elif menu == "ℹ️ About":

    st.title(
        "ℹ️ About EcoVision AI"
    )

    st.write(
        """
        EcoVision AI menggunakan Computer Vision dan
        Artificial Intelligence untuk membantu mengenali
        jenis sampah secara otomatis.
        """
    )

    st.markdown(
        """
        ### Teknologi

        - Python
        - Streamlit
        - OpenCV
        - Roboflow
        - YOLO
        - Computer Vision
        """
    )

    st.success(
        "♻️ Sort waste. Save the planet."
    )