import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
# ===========================
# Konfigurasi halaman
# ===========================
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="♻️"
)
st.write("🚨 CAMERA.PY BERHASIL DIBUKA")
st.write("Kalau tulisan ini muncul berarti Streamlit menjalankan camera.py")

st.title("♻️ EcoVision AI")
st.write("Upload gambar sampah untuk diklasifikasikan.")

# ===========================
# Nama kelas
# ===========================
classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    6
)

model.load_state_dict(
    torch.load(
        "models/ecovision_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ===========================
# Transform gambar
# ===========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        tensor = transform(pil).unsqueeze(0)

        with torch.no_grad():

            output = model(tensor)

            probs = torch.softmax(output, dim=1)

            confidence, predicted = torch.max(probs, 1)

            label = classes[predicted.item()]

            conf = confidence.item() * 100

        cv2.putText(
            img,
            f"{label} ({conf:.1f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )
# ===========================
# Upload gambar
# ===========================
st.header("📷 Live Camera")

webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
        },
        "audio": False,
    },
)