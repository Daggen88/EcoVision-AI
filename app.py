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
# RUN ROBOFLOW INFERENCE
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

    print("========== WORKFLOW RESULT ==========")
    print(result)
    print("=====================================")

    return result


def extract_predictions(result):
    """
    Mengambil predictions dari output Roboflow Workflow.
    """

    try:

        # ====================================================
        # OUTPUT WORKFLOW
        # ====================================================

        outputs = result.get("outputs", [])

        if not outputs:
            return []

        # Workflow menghasilkan list outputs
        first_output = outputs[0]

        if not isinstance(first_output, dict):
            return []

        # ====================================================
        # AMBIL predictions
        # ====================================================

        prediction_data = first_output.get(
            "predictions",
            {}
        )

        if not isinstance(prediction_data, dict):
            return []

        predictions = prediction_data.get(
            "predictions",
            []
        )

        if not isinstance(predictions, list):
            return []

        return predictions

    except Exception as e:

        print(
            "ERROR EXTRACT PREDICTIONS:",
            e
        )

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

    result_image = image.copy()

    for prediction in predictions:

        try:

            label = str(
                prediction.get(
                    "class",
                    "unknown"
                )
            )

            confidence = float(
                prediction.get(
                    "confidence",
                    0
                )
            )

            x = float(
                prediction.get(
                    "x",
                    0
                )
            )

            y = float(
                prediction.get(
                    "y",
                    0
                )
            )

            width = float(
                prediction.get(
                    "width",
                    0
                )
            )

            height = float(
                prediction.get(
                    "height",
                    0
                )
            )

            # Bounding box
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)

            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Pastikan tidak keluar gambar
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                result_image.shape[1],
                x2
            )

            y2 = min(
                result_image.shape[0],
                y2
            )

            # Bounding box
            cv2.rectangle(
                result_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Label
            text = (
                f"{label} "
                f"{confidence * 100:.1f}%"
            )

            cv2.putText(
                result_image,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        except Exception:
            continue

    return result_image


# ============================================================
# DETECTION INFORMATION
# ============================================================

def show_detection_metrics(predictions):

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

        return


    # Prediction dengan confidence tertinggi
    best_prediction = max(
        predictions,
        key=lambda p: float(
            p.get(
                "confidence",
                0
            )
        )
    )

    label = str(
        best_prediction.get(
            "class",
            "Unknown"
        )
    )

    confidence = float(
        best_prediction.get(
            "confidence",
            0
        )
    )

    category, recommendation = get_category(
        label
    )

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
            str(len(predictions))
        )

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
        EcoVision AI adalah aplikasi berbasis
        Computer Vision untuk mendeteksi jenis
        sampah menggunakan kamera atau gambar.
        """
    )

    st.markdown(
        """
        ### Fitur

        📷 **Live Detection**  
        Deteksi sampah menggunakan kamera.

        🖼️ **Image Detection**  
        Upload gambar sampah untuk dideteksi.

        ♻️ **Waste Classification**  
        Mengelompokkan sampah menjadi:
        - ORGANIK
        - ANORGANIK
        - B3
        - GENERAL WASTE
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

    camera_image = st.camera_input(
        "📸 Ambil gambar menggunakan kamera"
    )

    if camera_image is not None:

        # ====================================================
        # READ CAMERA IMAGE
        # ====================================================

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
                "❌ Gambar kamera tidak dapat dibaca."
            )

        else:

            # =================================================
            # INFERENCE
            # =================================================

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    workflow_result = run_inference(
                        image
                    )

                    predictions = extract_predictions(
                        workflow_result
                    )

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []


            # =================================================
            # ANNOTATED IMAGE
            # =================================================

            annotated_image = draw_predictions(
                image,
                predictions
            )


            # =================================================
            # DETECTION INFORMATION
            # =================================================

            st.subheader(
                "🔍 Detection Information"
            )

            show_detection_metrics(
                predictions
            )


            # =================================================
            # RESULT
            # =================================================

            if predictions:

                st.success(
                    "✅ Objek berhasil dideteksi."
                )

                st.image(
                    cv2.cvtColor(
                        annotated_image,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )

            else:

                st.warning(
                    "⚠️ Tidak ada objek yang terdeteksi."
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

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    workflow_result = run_inference(
                        image
                    )

                    predictions = extract_predictions(
                        workflow_result
                    )

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []


            annotated_image = draw_predictions(
                image,
                predictions
            )


            st.subheader(
                "🔍 Detection Information"
            )

            show_detection_metrics(
                predictions
            )


            if predictions:

                st.subheader(
                    "Detection Result"
                )

                st.image(
                    cv2.cvtColor(
                        annotated_image,
                        cv2.COLOR_BGR2RGB
                    ),
                    width="stretch"
                )

            else:

                st.warning(
                    "⚠️ Tidak ada objek yang terdeteksi."
                )


# ============================================================
# ABOUT
# ============================================================

elif menu == "ℹ️ About":

    st.title(
        "ℹ️ About EcoVision AI"
    )

    st.markdown(
        """
        ## ♻️ EcoVision AI

        EcoVision AI merupakan aplikasi
        Computer Vision yang menggunakan
        model YOLO melalui Roboflow Workflow
        untuk mendeteksi jenis sampah.

        ### Teknologi

        - Python
        - Streamlit
        - OpenCV
        - NumPy
        - Roboflow
        - YOLO

        ### Tujuan

        Membantu pengguna mengenali jenis sampah
        dan memberikan rekomendasi tempat
        pembuangan yang sesuai.
        """
    )