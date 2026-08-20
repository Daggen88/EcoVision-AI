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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ================================
       GENERAL
    ================================= */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ================================
       CAMERA INPUT
    ================================= */

    /* Sembunyikan gambar preview camera
       setelah foto diambil */
    div[data-testid="stCameraInput"] img {
        display: none !important;
    }

    /* ================================
       DETECTION CARD
    ================================= */

    .detection-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ROBOFLOW CONFIG
# ============================================================

load_dotenv()

# Local PC -> .env
api_key = os.getenv("ROBOFLOW_API_KEY")

# Streamlit Cloud -> Secrets
if not api_key:
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        api_key = None


if not api_key:

    st.error(
        "❌ ROBOFLOW_API_KEY belum dikonfigurasi.\n\n"
        "Untuk Streamlit Cloud, masukkan API key "
        "di App Settings → Secrets."
    )

    st.stop()


# ============================================================
# ROBOFLOW CLIENT
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)


# ============================================================
# WORKFLOW CONFIG
# ============================================================

WORKSPACE_NAME = "daggen580-gmail-com"

WORKFLOW_ID = (
    "my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic"
)


# ============================================================
# ROBOFLOW INFERENCE
# ============================================================

def run_inference(image):
    """
    Menjalankan Roboflow Workflow
    menggunakan gambar OpenCV.
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

def extract_predictions(result):
    """
    Mengambil list predictions dari hasil
    Roboflow Workflow.

    Struktur hasil Roboflow bisa berbeda,
    jadi fungsi ini dibuat fleksibel.
    """

    if result is None:
        return []


    # --------------------------------------------------------
    # Kalau result langsung berupa list
    # --------------------------------------------------------

    if isinstance(result, list):

        return [
            item
            for item in result
            if isinstance(item, dict)
        ]


    # --------------------------------------------------------
    # Kalau result berupa dictionary
    # --------------------------------------------------------

    if isinstance(result, dict):

        # Case:
        # {
        #   "predictions": [...]
        # }

        predictions = result.get("predictions")

        if isinstance(predictions, list):

            return predictions


        # Case:
        # {
        #   "outputs": {
        #       "predictions": {
        #           "predictions": [...]
        #       }
        #   }
        # }

        outputs = result.get("outputs")

        if isinstance(outputs, dict):

            prediction_output = outputs.get(
                "predictions"
            )

            if isinstance(
                prediction_output,
                dict
            ):

                predictions = prediction_output.get(
                    "predictions"
                )

                if isinstance(
                    predictions,
                    list
                ):

                    return predictions


            if isinstance(
                prediction_output,
                list
            ):

                return prediction_output


    return []


# ============================================================
# CATEGORY
# ============================================================

def get_category(label):
    """
    Menentukan kategori sampah.
    """

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
    """
    Menggambar bounding box dan label
    pada gambar.
    """

    output = image.copy()


    for prediction in predictions:

        if not isinstance(
            prediction,
            dict
        ):
            continue


        # ----------------------------------------------------
        # Get values
        # ----------------------------------------------------

        label = prediction.get(
            "class",
            prediction.get(
                "label",
                "Unknown"
            )
        )

        confidence = prediction.get(
            "confidence",
            0
        )


        try:

            confidence = float(
                confidence
            )

        except:

            confidence = 0


        # Bounding box Roboflow:
        # x, y = center
        # width, height = ukuran box

        try:

            x = float(
                prediction.get("x", 0)
            )

            y = float(
                prediction.get("y", 0)
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

        except:

            continue


        # ----------------------------------------------------
        # Convert center -> corner
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
        # Keep inside image
        # ----------------------------------------------------

        h, w = output.shape[:2]

        x1 = max(
            0,
            min(x1, w - 1)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        x2 = max(
            0,
            min(x2, w - 1)
        )

        y2 = max(
            0,
            min(y2, h - 1)
        )


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text = (
            f"{label} "
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


        # Background label

        cv2.rectangle(
            output,
            (
                x1,
                max(
                    0,
                    y1 - text_height - 12
                )
            ),
            (
                x1 + text_width + 10,
                y1
            ),
            (0, 255, 0),
            -1
        )


        # Text

        cv2.putText(
            output,
            text,
            (
                x1 + 5,
                y1 - 6
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )


    return output


# ============================================================
# DETECTION METRICS
# ============================================================

def show_detection_metrics(predictions):

    if not predictions:

        st.info(
            "🔍 Tidak ada objek yang terdeteksi."
        )

        return


    # --------------------------------------------------------
    # Ambil prediction dengan confidence tertinggi
    # --------------------------------------------------------

    best_prediction = max(
        predictions,
        key=lambda x: float(
            x.get(
                "confidence",
                0
            )
        )
        if isinstance(x, dict)
        else 0
    )


    label = best_prediction.get(
        "class",
        best_prediction.get(
            "label",
            "Unknown"
        )
    )


    try:

        confidence = float(
            best_prediction.get(
                "confidence",
                0
            )
        )

    except:

        confidence = 0


    category, recommendation = get_category(
        label
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Detected Object",
            str(label).upper()
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
            len(predictions)
        )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.info(
        f"♻️ Rekomendasi: {recommendation}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "♻️ EcoVision AI"
    )


    st.write(
        "### Navigation"
    )


    menu = st.radio(
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
        "AI Waste Detection System"
    )


    st.write(
        """
        EcoVision AI adalah aplikasi computer vision
        untuk mendeteksi jenis sampah menggunakan
        kamera atau gambar.
        """
    )


    st.info(
        """
        📷 Gunakan **Live Detection** untuk mengambil
        gambar menggunakan kamera.

        🖼️ Gunakan **Image Detection** untuk meng-upload
        gambar sampah.
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
        "Gunakan kamera perangkat untuk mengambil gambar "
        "dan mendeteksi jenis sampah."
    )


    # ========================================================
    # CAMERA
    # ========================================================

    camera_image = st.camera_input(
        "Ambil gambar menggunakan kamera"
    )


    if camera_image is not None:


        # ====================================================
        # READ IMAGE
        # ====================================================

        file_bytes = np.asarray(
            bytearray(
                camera_image.read()
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

            # =================================================
            # INFERENCE
            # =================================================

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    result = run_inference(
                        image
                    )


                    predictions = extract_predictions(
                        result
                    )


                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )


                    predictions = []


            # =================================================
            # DRAW RESULT
            # =================================================

            annotated_image = draw_predictions(
                image,
                predictions
            )


            # =================================================
            # DETECTION RESULT
            # =================================================

            st.subheader(
                "🔍 Detection Result"
            )


            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )


            # =================================================
            # INFORMATION
            # =================================================

            st.subheader(
                "🔎 Detection Information"
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


        # ====================================================
        # READ FILE
        # ====================================================

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

            # =================================================
            # INFERENCE
            # =================================================

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    result = run_inference(
                        image
                    )


                    predictions = extract_predictions(
                        result
                    )


                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )


                    predictions = []


            # =================================================
            # RESULT
            # =================================================

            annotated_image = draw_predictions(
                image,
                predictions
            )


            st.subheader(
                "🔍 Detection Result"
            )


            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )


            # =================================================
            # INFORMATION
            # =================================================

            st.subheader(
                "🔎 Detection Information"
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
        ### ♻️ EcoVision AI

        EcoVision AI adalah aplikasi berbasis
        Artificial Intelligence dan Computer Vision
        yang digunakan untuk mendeteksi jenis sampah.

        Sistem menggunakan model YOLO yang telah
        dilatih untuk mengenali berbagai jenis sampah.
        """
    )


    st.markdown(
        """
        ### Fitur

        📷 **Live Detection**  
        Mendeteksi sampah menggunakan kamera.

        🖼️ **Image Detection**  
        Mendeteksi sampah dari gambar yang di-upload.

        🔎 **Object Detection**  
        Menampilkan bounding box dan confidence.

        ♻️ **Waste Classification**  
        Mengelompokkan sampah menjadi:

        - ORGANIK
        - ANORGANIK
        - B3
        - GENERAL WASTE
        """
    )


    st.success(
        "🌱 EcoVision AI — Smart Waste Detection"
    )