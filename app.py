import streamlit as st
import cv2
import time
import numpy as np
import os
import threading

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

# ==========================
# Konfigurasi Halaman
# ==========================

st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️",
    layout="wide"
)
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-10-2/weights/best.pt")

model = load_model()

# ==========================
# ROBOFLOW
# ==========================

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    st.error("ROBOFLOW_API_KEY tidak ditemukan di .env")
    st.stop()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

MODEL_ID = "daggen580-gmail-com/my-first-project-owy0e-4-yolo11s-t1"

# ==========================
# Sidebars
# ==========================

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

# ==========================
# HOME
# ==========================

if menu == "🏠 Home":

    st.title("♻️ EcoVision AI")

    st.markdown("""
### Realtime Waste Detection using YOLO11

Project AI Engineer Bootcamp

EcoVision AI mampu mendeteksi:

- 📦 Cardboard
- 🥛 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑 Trash
""")

    st.success("Model berhasil dimuat.")

elif menu == "📷 Live Detection":

    st.title("📷 Live Detection")

    st.info("Tekan ESC pada window camera untuk menghentikan camera.")

    # ==========================
    # SHARED VARIABLES
    # ==========================

    predictions = []
    latest_frame = None

    lock = threading.Lock()
    running = True

    # ==========================
    # ROBOFLOW INFERENCE THREAD
    # ==========================

    def inference_loop():

        global predictions
        global latest_frame
        global running

        while running:

            # ==========================
            # AMBIL FRAME TERBARU
            # ==========================

            with lock:

                if latest_frame is None:
                    time.sleep(0.01)
                    continue

                frame = latest_frame.copy()

            # ==========================
            # PASTIKAN UKURAN FRAME
            # ==========================

            frame = cv2.resize(
                frame,
                (640, 480)
            )

            # ==========================
            # ROBOFLOW INFERENCE
            # ==========================

            try:

                result = client.infer(
                    frame,
                    model_id=MODEL_ID
                )

                new_predictions = result.get(
                    "predictions",
                    []
                )

                # DEBUG
                print(
                    f"Roboflow detected: "
                    f"{len(new_predictions)} object(s)"
                )

                if new_predictions:
                    print(
                        "Object:",
                        new_predictions[0]["class"],
                        "| Confidence:",
                        f"{new_predictions[0]['confidence'] * 100:.1f}%"
                    )

                # ==========================
                # UPDATE PREDICTION
                # ==========================

                with lock:
                    predictions = new_predictions

            except Exception as e:

                print(
                    "Inference error:",
                    e
                )

            # Jangan terlalu sering request
            time.sleep(0.15)
    # ==========================
    # START THREAD
    # ==========================

    thread = threading.Thread(
        target=inference_loop,
        daemon=True
    )

    thread.start()

    # ==========================
    # STREAMLIT UI
    # ==========================

    st.subheader("🔍 Detection Information")

    col1, col2, col3, col4 = st.columns(4)

    object_metric = col1.empty()
    confidence_metric = col2.empty()
    category_metric = col3.empty()
    count_metric = col4.empty()

    FRAME_WINDOW = st.image([])

    # ==========================
    # CAMERA
    # ==========================

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():

        st.error("❌ Camera tidak dapat dibuka.")

        running = False

    else:

        prev_time = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:

                st.error("❌ Gagal membaca frame.")

                break

            # ==========================
            # SEND FRAME TO INFERENCE
            # ==========================

            with lock:

                latest_frame = frame.copy()
                current_predictions = predictions.copy()

            # ==========================
            # DEFAULT
            # ==========================

            detected_object = "No Object"
            detected_confidence = 0
            detected_category = "-"
            instruction = "Arahkan sampah ke kamera"

            # ==========================
            # DRAW PREDICTIONS
            # ==========================

            for prediction in current_predictions:

                label = prediction["class"]
                confidence = prediction["confidence"]

                x = prediction["x"]
                y = prediction["y"]

                width = prediction["width"]
                height = prediction["height"]

                x1 = int(x - width / 2)
                y1 = int(y - height / 2)

                x2 = int(x + width / 2)
                y2 = int(y + height / 2)

                # ==========================
                # CATEGORY
                # ==========================

                if label.lower() == "battery":

                    category = "B3"
                    instruction = "Buang ke tempat khusus B3"

                elif label.lower() in [
                    "cardboard",
                    "paper",
                    "plastic",
                    "metal",
                    "glass"
                ]:

                    category = "ANORGANIK"
                    instruction = "Buang ke tempat sampah ANORGANIK"

                elif label.lower() in [
                    "organic",
                    "food",
                    "food-waste",
                    "organic-waste"
                ]:

                    category = "ORGANIK"
                    instruction = "Buang ke tempat sampah ORGANIK"

                else:

                    category = "UNKNOWN"
                    instruction = "Belum dikenali"

                # ==========================
                # BOUNDING BOX
                # ==========================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Label di atas bounding box
                cv2.putText(
                    frame,
                    f"{label} {confidence * 100:.1f}%",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ==========================
            # BEST PREDICTION
            # ==========================

            if current_predictions:

                best_prediction = max(
                    current_predictions,
                    key=lambda p: p["confidence"]
                )

                detected_object = best_prediction["class"]

                detected_confidence = (
                    best_prediction["confidence"] * 100
                )

                label = detected_object.lower()

                if label == "battery":

                    detected_category = "B3"
                    instruction = "Buang ke tempat khusus B3"

                elif label in [
                    "cardboard",
                    "paper",
                    "plastic",
                    "metal",
                    "glass"
                ]:

                    detected_category = "ANORGANIK"
                    instruction = "Buang ke tempat sampah ANORGANIK"

                elif label in [
                    "organic",
                    "food",
                    "food-waste",
                    "organic-waste"
                ]:

                    detected_category = "ORGANIK"
                    instruction = "Buang ke tempat sampah ORGANIK"

                else:

                    detected_category = "UNKNOWN"
                    instruction = "Belum dikenali"

            # ==========================
            # FPS
            # ==========================

            current_time = time.time()

            fps = 1 / max(
                current_time - prev_time,
                0.001
            )

            prev_time = current_time

            # ==========================
            # TOP INFORMATION PANEL
            # ==========================

            cv2.rectangle(
                frame,
                (10, 10),
                (330, 120),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.0f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Object: {detected_object}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Category: {detected_category}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{detected_confidence:.1f}%",
                (20, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # ==========================
            # STREAMLIT METRICS
            # ==========================

            object_metric.metric(
                "Detected Object",
                detected_object
            )

            confidence_metric.metric(
                "Confidence",
                f"{detected_confidence:.1f}%"
            )

            category_metric.metric(
                "Category",
                detected_category
            )

            count_metric.metric(
                "Objects",
                len(current_predictions)
            )

            # ==========================
            # DISPLAY
            # ==========================

            FRAME_WINDOW.image(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )
            )

            # ==========================
            # ESC
            # ==========================

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # ==========================
        # CLEANUP
        # ==========================

        running = False

        cap.release()

        cv2.destroyAllWindows()

        print("EcoVision AI camera ditutup.")
# ==========================
# IMAGE DETECTION
# ==========================

elif menu == "🖼️ Image Detection":

    st.title("🖼️ Image Detection")

    uploaded_file = st.file_uploader(
        "Upload gambar sampah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Baca gambar
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        # YOLO Detection
        results = model.predict(
            source=image,
            conf=0.70,
            verbose=False
        )

        boxes = results[0].boxes

        # ==========================
        # Detection Information
        # ==========================

        detected_object = "No Object"
        detected_confidence = 0
        detected_category = "-"
        object_count = len(boxes)

        waste_type = {
            "cardboard": "Recycle",
            "glass": "Recycle",
            "metal": "Recycle",
            "paper": "Recycle",
            "plastic": "Recycle",
            "trash": "General Waste"
        }

        if object_count > 0:

            best_box = max(
                boxes,
                key=lambda box: float(box.conf[0])
            )

            cls = int(best_box.cls[0])
            conf = float(best_box.conf[0])

            label = model.names[cls]

            detected_object = label
            detected_confidence = conf * 100

            detected_category = waste_type.get(
                label.lower(),
                "Unknown"
            )

        # ==========================
        # Metrics
        # ==========================

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
            object_count
        )

        # ==========================
        # Show Detection
        # ==========================

        annotated_image = results[0].plot()

        st.image(
            cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Detection Result",
            use_container_width=True
        )


# ==========================
# ABOUT
# ==========================

elif menu == "ℹ️ About":

    st.title("About")

    st.write("""
Developer

**Darren Vallenskie Sharlivt**

Universitas Kwik Kian Gie

AI Engineer Bootcamp Batch-12
""")
