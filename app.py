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

# Local PC -> ambil dari .env
api_key = os.getenv("ROBOFLOW_API_KEY")

# Streamlit Cloud -> ambil dari Secrets
if not api_key:
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error(
        "❌ ROBOFLOW_API_KEY belum dikonfigurasi.\n\n"
        "Untuk Streamlit Cloud, masukkan API key di App Settings → Secrets."
    )
    st.stop()


client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)


MODEL_ID = "daggen580-gmail-com/my-first-project-owy0e-4-yolo11s-t1"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_inference(image):
    result = client.run_workflow(
        workspace_name="daggen580-gmail-com",
        workflow_id="my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic",
        images={
            "image": image
        }
    )

    print("========== WORKFLOW RESULT ==========")
    print(type(result))
    print(result)
    print("=====================================")

    return result


def get_category(label):
    """
    Menentukan kategori sampah.
    """

    label = label.lower()

    if label == "battery":
        return "B3", "Buang ke tempat khusus B3"

    elif label in [
        "cardboard",
        "paper",
        "plastic",
        "metal",
        "glass"
    ]:
        return "ANORGANIK", "Buang ke tempat sampah ANORGANIK"

    elif label in [
        "organic",
        "food",
        "food-waste",
        "organic-waste"
    ]:
        return "ORGANIK", "Buang ke tempat sampah ORGANIK"

    elif label == "trash":
        return "GENERAL WASTE", "Buang ke tempat sampah umum"

    else:
        return "UNKNOWN", "Belum dikenali"


def draw_predictions(image, predictions):
    """
    Menggambar bounding box dan label hasil detection.
    """

    output = image.copy()

    for prediction in predictions:

        label = prediction.get("class", "Unknown")
        confidence = float(prediction.get("confidence", 0))

        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))

        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)

        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Pastikan bounding box tidak keluar gambar
        h, w = output.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Label
        text = f"{label} {confidence * 100:.1f}%"

        cv2.putText(
            output,
            text,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


def get_best_prediction(predictions):
    """
    Mengambil object dengan confidence tertinggi.
    """

    if not predictions:
        return None

    return max(
        predictions,
        key=lambda p: float(p.get("confidence", 0))
    )


def show_detection_metrics(predictions):
    """
    Menampilkan informasi detection.
    """

    detected_object = "No Object"
    detected_confidence = 0
    detected_category = "-"
    instruction = "Arahkan sampah ke kamera"

    if predictions:

        best_prediction = get_best_prediction(predictions)

        detected_object = best_prediction.get(
            "class",
            "Unknown"
        )

        detected_confidence = (
            float(best_prediction.get("confidence", 0)) * 100
        )

        detected_category, instruction = get_category(
            detected_object
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Detected Object",
        detected_object
    )

    col2.metric(
        "Confidence",
        f"{detected_confidence:.1f}%"
    )

    col3.metric(
        "Category",
        detected_category
    )

    col4.metric(
        "Objects",
        len(predictions)
    )

    if predictions:
        st.success(f"💡 {instruction}")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("♻️ EcoVision AI")

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

    st.title("♻️ EcoVision AI")

    st.markdown(
        """
        ### Realtime Waste Detection using YOLO11

        **Project AI Engineer Bootcamp**

        EcoVision AI mampu mendeteksi:

        - 📦 Cardboard
        - 🥛 Glass
        - 🥫 Metal
        - 📄 Paper
        - 🧴 Plastic
        - 🗑️ Trash
        - 🔋 Battery
        """
    )

    st.success("✅ EcoVision AI siap digunakan.")

    st.info(
        "Model inference menggunakan YOLO11 melalui Roboflow API."
    )


# ============================================================
# LIVE / CAMERA DETECTION
# ============================================================

elif menu == "📷 Live Detection":

    st.title("📷 Live Detection")

    st.write(
        "Gunakan kamera perangkat untuk mengambil gambar "
        "dan mendeteksi jenis sampah."
    )

    st.info(
        "📸 Tekan tombol kamera untuk mengambil gambar."
    )

    camera_image = st.camera_input(
        "Ambil gambar menggunakan kamera"
    )

    if camera_image is not None:

        # Baca gambar
        file_bytes = np.asarray(
            bytearray(camera_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("❌ Gagal membaca gambar.")

        else:

            # Resize agar request lebih ringan
            image = cv2.resize(
                image,
                (640, 480)
            )

            with st.spinner("🔍 Mendeteksi objek..."):

                try:

                    predictions = run_inference(
                        image
                    )

                except Exception as e:

                    st.error(
                        f"❌ Roboflow inference gagal: {e}"
                    )

                    predictions = []

            # Metrics
            st.subheader("🔍 Detection Information")

            show_detection_metrics(
                predictions
            )

            # Bounding box
            annotated_image = draw_predictions(
                image,
                predictions
            )

            # Display
            st.image(
                cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Detection Result",
                use_container_width=True
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

        # Baca file
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
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

            # Tampilkan gambar original
            st.subheader("Original Image")

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            # Inference
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

            # Detection Information
            st.subheader(
                "🔍 Detection Information"
            )

            show_detection_metrics(
                predictions
            )

            # Annotated image
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
                use_container_width=True
            )


# ============================================================
# ABOUT
# ============================================================

elif menu == "ℹ️ About":

    st.title("About EcoVision AI")

    st.markdown(
        """
        ## ♻️ EcoVision AI

        EcoVision AI adalah aplikasi computer vision
        untuk mendeteksi dan mengklasifikasikan sampah
        menggunakan model YOLO11.

        ### Features

        - 📷 Camera Detection
        - 🖼️ Image Detection
        - 🎯 Object Detection
        - 📊 Confidence Score
        - ♻️ Waste Category Classification
        - ☁️ Roboflow API Integration

        ### Developer

        **Darren Vallenskie Sharlivt**

        Universitas Kwik Kian Gie

        AI Engineer Bootcamp Batch-12
        """
    )

    st.success(
        "🚀 EcoVision AI successfully deployed."
    )