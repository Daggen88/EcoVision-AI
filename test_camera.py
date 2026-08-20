import os
import cv2
import time
import threading

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient


# =========================================================
# 1. LOAD ENV
# =========================================================

load_dotenv()


# =========================================================
# 2. ROBOFLOW CLIENT
# =========================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)


# =========================================================
# 3. WASTE CATEGORY
# =========================================================

WASTE_CATEGORY = {

    # ANORGANIK
    "cardboard": "ANORGANIK",
    "plastic": "ANORGANIK",
    "paper": "ANORGANIK",

    # ORGANIK
    "food": "ORGANIK",
    "banana": "ORGANIK",
    "apple": "ORGANIK",

    # B3
    "battery": "B3",
    "electronic": "B3",
}


# =========================================================
# 4. CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka!")
    exit()
print(
    "Camera resolution:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
# Resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("===================================")
print("      EcoVision-AI STARTED")
print("===================================")
print("Tekan Q untuk keluar.")


# =========================================================
# 5. SHARED VARIABLES
# =========================================================

latest_predictions = []

inference_running = False
inference_start_time = 0

lock = threading.Lock()


# =========================================================
# 6. INFERENCE FUNCTION
# =========================================================

def run_inference(frame):

    global latest_predictions
    global inference_running
    global inference_start_time

    start_time = time.time()

    try:

        print("\n===================================")
        print("INFERENCE START")
        print("===================================")

        # ============================================
        # 1. RESIZE FRAME
        # ============================================

        frame_small = cv2.resize(
            frame,
            (640, 480)
        )

        # ============================================
        # 2. SAVE JPEG
        # ============================================

        temp_image = "camera_frame.jpg"

        cv2.imwrite(
            temp_image,
            frame_small,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                60
            ]
        )

        print("Image prepared.")

        # ============================================
        # 3. SEND TO ROBOFLOW
        # ============================================

        print("Sending to Roboflow...")

        result = client.run_workflow(

            workspace_name="daggen580-gmail-com",

            workflow_id=
            "my-first-project-vmy-first-project-owy0e-3-yolo11s-t1-logic",

            images={
                "image": temp_image
            },

            use_cache=False
        )

        # ============================================
        # 4. GET PREDICTIONS
        # ============================================

        predictions = (
            result[0]
            ["predictions"]
            ["predictions"]
        )

        # ============================================
        # 5. UPDATE RESULT
        # ============================================

        with lock:

            latest_predictions = predictions

        elapsed = time.time() - start_time

        print(
            f"Objects detected: "
            f"{len(predictions)}"
        )

        print(
            f"Inference time: "
            f"{elapsed:.2f}s"
        )

        print("===================================")

    except Exception as e:

        print(
            f"Inference error: {e}"
        )

    finally:

        elapsed = time.time() - start_time

        print(f"Inference finished in {elapsed:.2f}s")

        inference_running = False


# =========================================================
# 7. CAMERA LOOP
# =========================================================

last_inference = 0

inference_interval = 1.0


while True:

    # -----------------------------------------------------
    # READ CAMERA
    # -----------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Gagal mengambil frame.")
        break


    current_time = time.time()


    # =====================================================
    # START BACKGROUND INFERENCE
    # =====================================================

    if (
        current_time - last_inference >= inference_interval
        and not inference_running
    ):

        inference_running = True
        inference_start_time = current_time

        last_inference = current_time

        inference_frame = frame.copy()

        thread = threading.Thread(
            target=run_inference,
            args=(inference_frame,),
            daemon=True
        )

        thread.start()


    # =====================================================
    # GET LATEST PREDICTIONS
    # =====================================================

    with lock:

        predictions = latest_predictions.copy()


    # =====================================================
    # FIND BEST DETECTION
    # =====================================================

    valid_predictions = [

        p for p in predictions

        if p.get("confidence", 0) >= 0.40
    ]


    best_prediction = None

    if valid_predictions:

        best_prediction = max(

            valid_predictions,

            key=lambda p: p["confidence"]
        )


    # =====================================================
    # DRAW ALL DETECTIONS
    # =====================================================

    for prediction in valid_predictions:

        confidence = prediction["confidence"]

        x = prediction["x"]
        y = prediction["y"]

        width = prediction["width"]
        height = prediction["height"]

        class_name = prediction["class"].lower()


        # -------------------------------------------------
        # CENTER -> CORNER
        # -------------------------------------------------

        x1 = int(x - width / 2)
        y1 = int(y - height / 2)

        x2 = int(x + width / 2)
        y2 = int(y + height / 2)


        # -------------------------------------------------
        # KEEP BOX INSIDE FRAME
        # -------------------------------------------------

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)


        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        category = WASTE_CATEGORY.get(
            class_name,
            "UNKNOWN"
        )


        # -------------------------------------------------
        # BOX COLOR
        # -------------------------------------------------

        if category == "ORGANIK":

            box_color = (0, 255, 0)

        elif category == "ANORGANIK":

            box_color = (255, 255, 0)

        elif category == "B3":

            box_color = (0, 0, 255)

        else:

            box_color = (255, 255, 255)


        # -------------------------------------------------
        # DRAW BOX
        # -------------------------------------------------

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            box_color,

            2
        )


        # -------------------------------------------------
        # LABEL
        # -------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence * 100:.1f}%"
        )


        cv2.putText(

            frame,

            label,

            (
                x1,
                max(y1 - 10, 25)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            box_color,

            2
        )


    # =====================================================
    # UI HEADER
    # =====================================================

    cv2.putText(

        frame,

        "EcoVision-AI",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (0, 255, 0),

        2
    )


    # =====================================================
    # ANALYZING STATUS
    # =====================================================

    if inference_running:

        elapsed = time.time() - inference_start_time

        analyzing_text = f"Analyzing... {elapsed:.1f}s"

        cv2.putText(
            frame,
            analyzing_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # =====================================================
# RESULT UI
# =====================================================

    if best_prediction is not None:

        class_name = (
            best_prediction["class"]
            .lower()
        )

        confidence = best_prediction["confidence"]

        category = WASTE_CATEGORY.get(
            class_name,
            "UNKNOWN"
        )

        if category == "ORGANIK":
            destination = "TEMPAT SAMPAH ORGANIK"

        elif category == "ANORGANIK":
            destination = "TEMPAT SAMPAH ANORGANIK"

        elif category == "B3":
            destination = "TEMPAT SAMPAH B3"

        else:
            destination = "TIDAK DIKETAHUI"

        # =========================
        # RESULT PANEL
        # =========================

        cv2.rectangle(
            frame,
            (80, 130),
            (570, 280),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"OBJECT: {class_name.upper()}",
            (95, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"CONFIDENCE: {confidence * 100:.1f}%",
            (95, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"KATEGORI: {category}",
            (95, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"BUANG KE: {destination}",
            (95, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    elif not inference_running:

        cv2.putText(
            frame,
            "Tidak ada sampah terdeteksi",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(

        "EcoVision-AI",

        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# 8. CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("EcoVision-AI stopped.")