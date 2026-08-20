from inference_sdk import InferenceHTTPClient
import os
from dotenv import load_dotenv

print("1. Starting...")

# Load API key
load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

print("2. API key loaded:", bool(api_key))

# Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

print("3. Client ready")

# Model
MODEL_ID = "daggen580-gmail-com/my-first-project-owy0e-1-yolo11s-t1"

# Folder dataset
cardboard_folder = "Dataset/Custom/images/cardboard"
paper_folder = "Dataset/Custom/images/paper"


def test_folder(folder_path, expected_class):
    print("\n========================================")
    print(f"TESTING: {expected_class.upper()}")
    print("========================================")

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    print(f"Total images: {len(files)}")

    detected_correctly = 0
    no_detection = 0

    for i, filename in enumerate(files, start=1):
        image_path = os.path.join(folder_path, filename)

        try:
            result = client.infer(
                image_path,
                model_id=MODEL_ID
            )

            predictions = result.get("predictions", [])

            detected_classes = [
                prediction.get("class")
                for prediction in predictions
            ]

            if expected_class in detected_classes:
                detected_correctly += 1
                status = "OK"
            else:
                no_detection += 1
                status = "MISS"

            print(
                f"[{i}/{len(files)}] "
                f"{status} | "
                f"{filename} | "
                f"Detected: {detected_classes}"
            )

        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR | {filename}")
            print(e)

    print("\n----------------------------------------")
    print(f"{expected_class.upper()} SUMMARY")
    print("----------------------------------------")
    print(f"Total images      : {len(files)}")
    print(f"Detected correctly: {detected_correctly}")
    print(f"Missed            : {no_detection}")

    if len(files) > 0:
        accuracy = detected_correctly / len(files) * 100
        print(f"Detection rate    : {accuracy:.2f}%")
    else:
        print("No images found!")


# Test cardboard
test_folder(
    cardboard_folder,
    "cardboard"
)

# Test paper
test_folder(
    paper_folder,
    "paper"
)

print("\n========================================")
print("TEST FINISHED")
print("========================================")