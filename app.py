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
        "Masukkan API key di Streamlit Cloud → App Settings → Secrets."
    )
    st.stop()


# ============================================================
# ROBOFLOW WORKFLOW
# ============================================================

WORKSPACE_NAME = "daggen580-gmail-com"

WORKFLOW_ID = (
    "my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic"
)


client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)


# ============================================================
# RUN ROBOFLOW WORKFLOW
# ============================================================

def run_inference(image):
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={
            "image": image
        }
    )

    print("WORKFLOW RESULT:")
    print(result)

    # Roboflow workflow mengembalikan list
    if isinstance(result, list):
        if len(result) == 0:
            return []
        result = result[0]

    # Ambil outputs
    if not isinstance(result, dict):
        return []

    outputs = result.get("outputs", {})

    if not isinstance(outputs, dict):
        return []

    predictions = outputs.get("predictions", {})

    # Struktur yang biasanya keluar dari workflow
    if isinstance(predictions, dict):

        predictions = predictions.get(
            "predictions",
            []
        )

    if isinstance(predictions, list):
        return predictions

    return []

# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

def extract_predictions(data):
    """
    Roboflow Workflow dapat mengembalikan struktur
    dictionary/list yang berbeda.

    Fungsi ini mencari list prediction secara aman.
    """

    # --------------------------------------------------------
    # Kalau langsung list
    # --------------------------------------------------------

    if isinstance(data, list):

        # Kalau list berisi prediction langsung
        valid_predictions = []

        for item in data:

            if isinstance(item, dict):

                if (
                    "class" in item
                    or "confidence" in item
                ):
                    valid_predictions.append(item)

        if valid_predictions:
            return valid_predictions

        # Kalau list berisi object lain
        for item in data:

            result = extract_predictions(item)

            if result:
                return result

        return []


    # --------------------------------------------------------
    # Kalau dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        # Kasus paling umum:
        #
        # outputs
        #   └── predictions
        #         └── predictions
        #

        if "predictions" in data:

            predictions_data = data["predictions"]

            # predictions langsung list
            if isinstance(predictions_data, list):

                valid_predictions = []

                for item in predictions_data:

                    if isinstance(item, dict):

                        if (
                            "class" in item
                            or "confidence" in item
                        ):
                            valid_predictions.append(item)

                if valid_predictions:
                    return valid_predictions

            # predictions berupa dictionary
            result = extract_predictions(
                predictions_data
            )

            if result:
                return result


        # Cari di outputs
        if "outputs" in data:

            result = extract_predictions(
                data["outputs"]
            )

            if result:
                return result


        # Cari di semua value dictionary
        for value in data.values():

            if isinstance(value, (dict, list)):

                result = extract_predictions(
                    value
                )

                if result:
                    return result

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

    if not isinstance(predictions, list):
        return annotated

    for prediction in predictions:

        if not isinstance(prediction, dict):
            continue

        x = float(
            prediction.get("x", 0)
        )

        y = float(
            prediction.get("y", 0)
        )

        width = float(
            prediction.get("width", 0)
        )

        height = float(
            prediction.get("height", 0)
        )

        confidence = float(
            prediction.get("confidence", 0)
        )

        label = str(
            prediction.get(
                "class",
                "Unknown"
            )
        )

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

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        text = (
            f"{label} "
            f"{confidence * 100:.1f}%"
        )

        cv2.putText(
            annotated,
            text,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return annotated
        # ----------------------------------------------------
        # Convert center coordinates
        # ke bounding box
        # ----------------------------------------------------

        

# ============================================================
# DETECTION INFORMATION
# ============================================================

def show_detection_metrics(predictions):

    if not isinstance(predictions, list):
        predictions = []

    if len(predictions) == 0:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Detected Object",
            "No Object"
        )

        col2.metric(
            "Confidence",
            "0.0%"
        )

        col3.metric(
            "Category",
            "-"
        )

        col4.metric(
            "Objects",
            "0"
        )

        return

    best = max(
        predictions,
        key=lambda p: float(
            p.get("confidence", 0)
        )
        if isinstance(p, dict)
        else 0
    )

    label = best.get(
        "class",
        "Unknown"
    )

    confidence = float(
        best.get(
            "confidence",
            0
        )
    )

    category, recommendation = get_category(
        label
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Detected Object",
        str(label).upper()
    )

    col2.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )

    col3.metric(
        "Category",
        category
    )

    col4.metric(
        "Objects",
        str(len(predictions))
    )

    st.info(
        f"♻️ Rekomendasi: {recommendation}"
    )


    # --------------------------------------------------------
    # Ambil object dengan confidence tertinggi
    # --------------------------------------------------------

    best_prediction = max(
        predictions,
        key=lambda x: float(
            x.get(
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


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Detected Object",
        label.upper()
    )


    col2.metric(
        "Confidence",
        f"{confidence * 100:.1f}%"
    )


    col3.metric(
        "Category",
        category
    )


    col4.metric(
        "Objects",
        str(len(predictions))
    )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.info(
        f"♻️ Rekomendasi: {recommendation}"
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    try:

        with st.spinner(
            "🔍 Mendeteksi objek..."
        ):

            predictions = run_inference(
                image
            )

        return predictions

    except Exception as e:

        st.error(
            f"❌ Roboflow inference gagal: {e}"
        )

        print(
            "ROBOFLOW ERROR:",
            repr(e)
        )

        return []


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
        Artificial Intelligence untuk mendeteksi
        jenis sampah menggunakan computer vision.
        """
    )


    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            ### 📷 Live Detection

            Gunakan kamera perangkat untuk
            mengambil gambar dan mendeteksi
            jenis sampah.
            """
        )


    with col2:

        st.markdown(
            """
            ### 🖼️ Image Detection

            Upload gambar sampah dan biarkan
            AI melakukan deteksi otomatis.
            """
        )


    with col3:

        st.markdown(
            """
            ### ♻️ Smart Recommendation

            Sistem memberikan rekomendasi
            tempat pembuangan berdasarkan
            jenis sampah.
            """
        )


# ============================================================
# LIVE DETECTION
# ============================================================

elif menu == "📷 Live Detection":

    st.title("📷 Live Detection")

    st.write(
        "Gunakan kamera perangkat untuk mengambil gambar "
        "dan mendeteksi jenis sampah."
    )

    camera_image = st.camera_input(
        "📸 Ambil gambar menggunakan kamera"
    )

    if camera_image is not None:

        file_bytes = np.asarray(
            bytearray(camera_image.getvalue()),
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

            with st.spinner(
                "🔍 Mendeteksi objek..."
            ):

                try:

                    predictions = run_inference(image)

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []

            # Gambar hasil deteksi langsung di gambar kamera
            annotated_image = draw_predictions(
                image,
                predictions
            )

            # SATU GAMBAR SAJA
            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )

            # Informasi deteksi
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

    st.title("🖼️ Image Detection")

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
                    predictions = run_inference(
                        image
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
                "🔍 Detection Result"
            )

            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )

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


    st.markdown(
        """
        ## ♻️ EcoVision AI

        EcoVision AI merupakan aplikasi computer vision
        yang menggunakan Artificial Intelligence untuk
        membantu mengenali jenis sampah.

        ### Fitur

        - 📷 Live Detection
        - 🖼️ Image Detection
        - 🤖 AI Object Detection
        - ♻️ Waste Classification
        - 📊 Confidence Score
        - 🗑️ Waste Disposal Recommendation

        ### Teknologi

        - Python
        - Streamlit
        - OpenCV
        - NumPy
        - Roboflow
        - YOLO
        """
    )


    st.success(
        "EcoVision AI — Smart Waste Detection"
    )