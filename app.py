import streamlit as st
import cv2
import numpy as np
import os
import base64
import requests

from dotenv import load_dotenv


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


WORKSPACE_NAME = "daggen580-gmail-com"

WORKFLOW_ID = (
    "my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic"
)


# ============================================================
# ROBOFLOW WORKFLOW
# ============================================================

def run_inference(image):
    """
    Menjalankan Roboflow Workflow menggunakan HTTP API.
    """

    # --------------------------------------------------------
    # Encode OpenCV image -> JPG -> Base64
    # --------------------------------------------------------

    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise Exception(
            "Gagal mengubah gambar menjadi JPG."
        )

    image_base64 = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


    # --------------------------------------------------------
    # Roboflow Workflow endpoint
    # --------------------------------------------------------

    url = (
        "https://serverless.roboflow.com/"
        f"{WORKSPACE_NAME}/workflows/"
        f"{WORKFLOW_ID}"
    )


    # --------------------------------------------------------
    # Request body
    # --------------------------------------------------------

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": image_base64
            }
        }
    }


    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    response = requests.post(
        url,
        json=payload,
        timeout=60
    )


    # --------------------------------------------------------
    # Error handling
    # --------------------------------------------------------

    if response.status_code != 200:

        raise Exception(
            f"Roboflow API Error "
            f"{response.status_code}: "
            f"{response.text}"
        )


    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    result = response.json()

    print("========== WORKFLOW RESULT ==========")
    print(result)
    print("=====================================")

    return result


# ============================================================
# EXTRACT PREDICTIONS
# ============================================================

def extract_predictions(result):
    """
    Mengambil predictions dari berbagai kemungkinan
    struktur output Roboflow Workflow.
    """

    if result is None:
        return []


    # --------------------------------------------------------
    # Case 1:
    #
    # {
    #     "outputs": [
    #         {
    #             "predictions": [...]
    #         }
    #     ]
    # }
    # --------------------------------------------------------

    if isinstance(result, dict):

        outputs = result.get("outputs")

        if isinstance(outputs, list):

            for output in outputs:

                if not isinstance(output, dict):
                    continue

                if "predictions" in output:

                    predictions = output["predictions"]

                    if isinstance(predictions, list):
                        return predictions


        # ----------------------------------------------------
        # Direct predictions
        # ----------------------------------------------------

        if "predictions" in result:

            predictions = result["predictions"]

            if isinstance(predictions, list):
                return predictions


    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    def search(obj):

        if isinstance(obj, dict):

            if "predictions" in obj:

                value = obj["predictions"]

                if isinstance(value, list):
                    return value


            for value in obj.values():

                found = search(value)

                if found is not None:
                    return found


        elif isinstance(obj, list):

            for item in obj:

                found = search(item)

                if found is not None:
                    return found


        return None


    found = search(result)

    if found is not None:
        return found


    return []


# ============================================================
# CATEGORY
# ============================================================

def get_category(label):

    if not label:
        return (
            "UNKNOWN",
            "Belum dikenali"
        )


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


    return (
        "UNKNOWN",
        "Belum dikenali"
    )


# ============================================================
# DRAW PREDICTIONS
# ============================================================

def draw_predictions(image, predictions):

    result = image.copy()


    if not predictions:
        return result


    for prediction in predictions:

        try:

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

            label = prediction.get(
                "class",
                prediction.get(
                    "label",
                    "unknown"
                )
            )


            # ------------------------------------------------
            # Convert center coordinates
            # -> top-left / bottom-right
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            h, w = result.shape[:2]

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


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            text = (
                f"{label} "
                f"{confidence * 100:.1f}%"
            )


            cv2.putText(
                result,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        except Exception as e:

            print(
                "Prediction drawing error:",
                e
            )


    return result


# ============================================================
# DETECTION METRICS
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


    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    valid_predictions = []

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict
        ):
            continue

        valid_predictions.append(
            prediction
        )


    if not valid_predictions:

        st.warning(
            "⚠️ Tidak ada prediction yang valid."
        )

        return


    best = max(
        valid_predictions,
        key=lambda p: float(
            p.get(
                "confidence",
                0
            )
        )
    )


    label = best.get(
        "class",
        best.get(
            "label",
            "Unknown"
        )
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
            str(len(valid_predictions))
        )


    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.info(
        f"♻️ **Rekomendasi:** {recommendation}"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "♻️ EcoVision AI"
)


menu = st.sidebar.radio(
    "Navigation",
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
        model AI dari Roboflow.
        """
    )


    st.divider()


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            "📷 **Camera Detection**\n\n"
            "Gunakan kamera perangkat."
        )


    with col2:

        st.info(
            "🖼️ **Image Detection**\n\n"
            "Upload gambar sampah."
        )


    with col3:

        st.info(
            "♻️ **Waste Classification**\n\n"
            "Dapatkan kategori dan rekomendasi."
        )


# ============================================================
# LIVE DETECTION
# ============================================================

elif menu == "📷 Live Detection":

    st.title(
        "📷 Live Detection"
    )


    st.write(
        "Gunakan kamera perangkat untuk "
        "mengambil gambar dan mendeteksi sampah."
    )


    # ========================================================
    # CAMERA
    # ========================================================

    camera_image = st.camera_input(
        "Ambil gambar menggunakan kamera"
    )


    if camera_image is not None:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Detection Information
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Information"
            )


            show_detection_metrics(
                predictions
            )


            # ------------------------------------------------
            # ONE DETECTION IMAGE
            # ------------------------------------------------

            annotated_image = draw_predictions(
                image,
                predictions
            )


            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
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
        # Read image
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
            # Original
            # ------------------------------------------------

            st.subheader(
                "Original Image"
            )


            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )


            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Detection Information
            # ------------------------------------------------

            st.subheader(
                "🔍 Detection Information"
            )


            show_detection_metrics(
                predictions
            )


            # ------------------------------------------------
            # Detection Result
            # ------------------------------------------------

            annotated_image = draw_predictions(
                image,
                predictions
            )


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


            # ------------------------------------------------
            # Raw predictions
            # ------------------------------------------------

            if predictions:

                with st.expander(
                    "🔎 View Detection Data"
                ):

                    st.json(
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
        **EcoVision AI** adalah aplikasi
        computer vision untuk membantu mengenali
        jenis sampah secara otomatis.
        """
    )


    st.divider()


    st.subheader(
        "🤖 Technology"
    )


    st.write(
        """
        - Python
        - Streamlit
        - OpenCV
        - NumPy
        - Roboflow
        - YOLO
        """
    )


    st.subheader(
        "♻️ Waste Categories"
    )


    st.write(
        """
        **ORGANIK**  
        Sampah organik seperti food waste.

        **ANORGANIK**  
        Plastic, paper, cardboard, glass, metal.

        **B3**  
        Contohnya battery.

        **GENERAL WASTE**  
        Sampah umum yang belum masuk kategori khusus.
        """
    )