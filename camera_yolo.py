from inference_sdk import InferenceHTTPClient
import cv2
import os
import time
import threading
from dotenv import load_dotenv


# ==========================
# LOAD ENV
# ==========================

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY tidak ditemukan di .env")


# ==========================
# ROBOFLOW CLIENT
# ==========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

MODEL_ID = "daggen580-gmail-com/my-first-project-owy0e-4-yolo11s-t1"


# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Webcam tidak bisa dibuka")


# ==========================
# SHARED VARIABLES
# ==========================

predictions = []
latest_frame = None

lock = threading.Lock()

running = True


# ==========================
# ROBOFLOW THREAD
# ==========================

def inference_loop():

    global predictions
    global latest_frame
    global running

    while running:

        with lock:

            if latest_frame is None:
                time.sleep(0.01)
                continue

            frame = latest_frame.copy()

        try:

            result = client.infer(
                frame,
                model_id=MODEL_ID
            )

            new_predictions = result.get(
                "predictions",
                []
            )

            with lock:
                predictions = new_predictions

            print(
                f"Detected: {len(new_predictions)} object(s)"
            )

        except Exception as e:

            print("Inference error:", e)

        # Jangan terlalu banyak request
        time.sleep(0.5)


# ==========================
# START THREAD
# ==========================

thread = threading.Thread(
    target=inference_loop,
    daemon=True
)

thread.start()


# ==========================
# MAIN WEBCAM LOOP
# ==========================

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:

        print("Gagal mengambil frame")
        break


    # ==========================
    # UPDATE FRAME
    # ==========================

    with lock:

        latest_frame = frame.copy()

        current_predictions = predictions.copy()


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


        # Center → corner

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)

        x2 = int(x + width / 2)
        y2 = int(y + height / 2)


        # ==========================
        # BOUNDING BOX
        # ==========================

        color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )


        # Label di bounding box

        cv2.putText(
            frame,
            f"{label} {confidence * 100:.1f}%",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # ==========================
    # STATUS PANEL
    # ==========================

    if current_predictions:

        # Ambil confidence tertinggi

        best_prediction = max(
            current_predictions,
            key=lambda p: p["confidence"]
        )

        label = best_prediction["class"]

        confidence = best_prediction["confidence"]


        # ==========================
        # WASTE CATEGORY
        # ==========================

        if label == "battery":

            category = "B3"
            instruction = "Buang ke tempat khusus B3"

        elif label in [
            "cardboard",
            "paper",
            "plastic",
            "metal",
            "glass"
        ]:

            category = "ANORGANIK"
            instruction = "Buang ke tempat sampah ANORGANIK"

        else:

            category = "UNKNOWN"
            instruction = "Belum dikenali"


        # ==========================
        # PANEL BACKGROUND
        # ==========================

        cv2.rectangle(
            frame,
            (20, 100),
            (620, 220),
            (0, 0, 0),
            -1
        )


        # ==========================
        # OBJECT
        # ==========================

        cv2.putText(
            frame,
            f"Object: {label.upper()} ({confidence * 100:.1f}%)",
            (35, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ==========================
        # CATEGORY
        # ==========================

        cv2.putText(
            frame,
            f"Kategori: {category}",
            (35, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ==========================
        # INSTRUCTION
        # ==========================

        cv2.putText(
            frame,
            instruction,
            (35, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


    # ==========================
    # FPS
    # ==========================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ==========================
    # TITLE
    # ==========================

    cv2.putText(
        frame,
        "EcoVision AI",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3
    )


    # ==========================
    # SHOW WEBCAM
    # ==========================

    cv2.imshow(
        "EcoVision AI",
        frame
    )


    # ==========================
    # ESC TO EXIT
    # ==========================

    if cv2.waitKey(1) & 0xFF == 27:

        break


# ==========================
# CLEANUP
# ==========================

running = False

cap.release()

cv2.destroyAllWindows()

print("EcoVision AI ditutup.")