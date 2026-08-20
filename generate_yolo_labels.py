import os
print("Current Folder:", os.getcwd())
import cv2
import shutil

# ======================
# PATH
# ======================

DATASET_ROOT = "Dataset/Garbage classification/Garbage classification"

SPLIT_ROOT = "Dataset"

YOLO = "Dataset/YOLO"

CLASS_MAP = {
    1: "glass",
    2: "paper",
    3: "cardboard",
    4: "plastic",
    5: "metal",
    6: "trash"
}

CLASS_ID = {
    "glass":0,
    "paper":1,
    "cardboard":2,
    "plastic":3,
    "metal":4,
    "trash":5
}

splits = {
    "train":"one-indexed-files-notrash_train.txt",
    "val":"one-indexed-files-notrash_val.txt",
    "test":"one-indexed-files-notrash_test.txt"
}


for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(YOLO, "images", split), exist_ok=True)
    os.makedirs(os.path.join(YOLO, "labels", split), exist_ok=True)

total = 0

for split,filelist in splits.items():

    with open(
        os.path.join(SPLIT_ROOT,filelist),
        "r"
    ) as f:

        lines = f.readlines()

    for line in lines:

        filename,classid = line.strip().split()

        classid = int(classid)

        classname = CLASS_MAP[classid]

        src = os.path.join(
            DATASET_ROOT,
            classname,
            filename
        )

        dst = os.path.join(
            YOLO,
            "images",
            split,
            filename
        )
        print(src)
        print(os.path.exists(src))
        print(src)
        print(os.path.exists(src))
        shutil.copy(src,dst)

        img = cv2.imread(src)

        if img is None:
            print(f"Gagal membaca: {src}")
            continue

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        _,th = cv2.threshold(
            gray,
            235,
            255,
            cv2.THRESH_BINARY_INV
        )

        cnts,_ = cv2.findContours(
            th,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(cnts)==0:
            continue

        c=max(cnts,key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(c)

        H,W = img.shape[:2]

        xc=(x+w/2)/W
        yc=(y+h/2)/H
        ww=w/W
        hh=h/H

        label=os.path.join(
            YOLO,
            "labels",
            split,
            filename.replace(".jpg",".txt")
        )

        with open(label,"w") as f:

            f.write(
                f"{CLASS_ID[classname]} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{ww:.6f} "
                f"{hh:.6f}"
            )
        total += 1 
print("="*40)
print("SELESAI")
print(f"Total gambar: {total}")
print("="*40)