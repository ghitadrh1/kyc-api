from ultralytics import YOLO
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
import os

# Charger les deux modèles YOLO
model_50 = YOLO("https://huggingface.co/ghitadrh/kyc-models/resolve/main/yolo_model.pt")
model_10 = YOLO("https://huggingface.co/ghitadrh/kyc-models/resolve/main/best.pt")

# Charger PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"page_{i + 1}.jpg"
        image.save(image_path, "JPEG")
        image_paths.append(image_path)
    return image_paths

def clean_address(text):
    text = text.replace('\n', ' ')
    match = re.search(r'[A-Z][^\n]*', text)
    return match.group().strip() if match else text.strip()

def clean_dob(text):
    cleaned_text = re.sub(r'[^0-9/]', '', text.replace('.', '/'))
    return cleaned_text.rstrip('/')

def detect_with_model(img, model):
    results = model(img)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        confidence = box.conf[0].item()
        if confidence < 0.3:
            continue
        x1 = max(int(box.xyxy[0][0]) - 10, 0)
        y1 = max(int(box.xyxy[0][1]) - 10, 0)
        x2 = min(int(box.xyxy[0][2]) + 10, img.shape[1])
        y2 = min(int(box.xyxy[0][3]) + 10, img.shape[0])
        cropped_img = img[y1:y2, x1:x2]
        ocr_result = ocr.ocr(cropped_img, cls=True)
        detected_text = [word[1][0] for line in ocr_result for word in line]
        detected_text_str = " ".join(detected_text)
        if class_name == "adresse":
            detected_text_str = clean_address(detected_text_str)
        elif class_name == "dob":
            detected_text_str = clean_dob(detected_text_str)
        detections.append({
            "class": class_name,
            "text": detected_text_str,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })
    return detections

def extract_fields_cin(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    image_paths = []

    if ext == ".pdf":
        image_paths = pdf_to_images(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        image_paths = [file_path]
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or image file.")

    extracted = {}
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image {image_path} not found.")

        detections_10 = detect_with_model(img, model_10)
        detections_50 = detect_with_model(img, model_50)

        required_classes = ["nom", "prenom", "dob", "id", "adresse"]
        all_detections = detections_10 + detections_50
        fused_results = {}
        for field in required_classes:
            candidates = [p for p in all_detections if p["class"] == field]
            if candidates:
                best = max(candidates, key=lambda x: x["confidence"])
                fused_results[field] = best["text"]
            else:
                fused_results[field] = ""

    return fused_results