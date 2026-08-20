import os
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# =========================
# 1. LOAD ENV
# =========================

load_dotenv()

# =========================
# 2. CONNECT TO ROBOFLOW
# =========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# =========================
# 3. IMAGE
# =========================

image_path = "test.jpg"

# Baca gambar
image = cv2.imread(image_path)

if image is None:
    print(f"Gambar tidak ditemukan: {image_path}")
    exit()

# =========================
# 4. RUN MODEL
# =========================

result = client.run_workflow(
    workspace_name="daggen580-gmail-com",
    workflow_id="my-first-project-vmy-first-project-owy0e-4-yolo11s-t1-logic",
    images={
        "image": image_path
    },
    use_cache=True
)

print(result)

# =========================
# 5. GET PREDICTIONS
# =========================

predictions = result[0]["predictions"]["predictions"]   

print(f"\nJumlah object terdeteksi: {len(predictions)}\n")

# =========================
# 6. DRAW BOUNDING BOX
# =========================

for prediction in predictions:

    x = prediction["x"]
    y = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]

    confidence = prediction["confidence"]
    class_name = prediction["class"]

    # Convert center coordinate -> corner coordinate
    x1 = int(x - width / 2)
    y1 = int(y - height / 2)

    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    # Bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    label = f"{class_name} {confidence * 100:.1f}%"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    print(
        f"{class_name}: "
        f"{confidence * 100:.1f}%"
    )

# =========================
# 7. SAVE RESULT
# =========================

output_path = "result.jpg"

cv2.imwrite(output_path, image)

print(f"\nHasil disimpan sebagai: {output_path}")

# =========================
# 8. SHOW IMAGE
# =========================

cv2.imshow("EcoVision-AI", image)

cv2.waitKey(0)
cv2.destroyAllWindows()