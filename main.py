from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import shutil
import os
import torch
import requests
from torchvision import models, transforms
from pdf2image import convert_from_path

from rc_extractor import predict_from_file  # RC
from cin_extractor import extract_fields_cin   # CIN
from rib_extractor import extract_fields_rib   # RIB

app = FastAPI()

# 🔹 Télécharger et charger le modèle de classification (ResNet)
classifier_url = "https://huggingface.co/ghitadrh/kyc-models/resolve/main/doc_classifier.pth"
local_path = "/tmp/doc_classifier.pth"

if not os.path.exists(local_path):
    with open(local_path, "wb") as f:
        f.write(requests.get(classifier_url).content)

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(local_path, map_location="cpu"))
model.eval()

label_map = {0: "cin", 1: "rc", 2: "rib"}

doc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_document(image: Image.Image) -> str:
    input_tensor = doc_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
    return label_map[pred_class]

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    merged_result = {}

    for file in files:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            if file.filename.lower().endswith(".pdf"):
                pages = convert_from_path(temp_path, dpi=300)

                for i, page in enumerate(pages):
                    doc_type = classify_document(page)

                    temp_image_path = f"/tmp/{file.filename}_page_{i+1}.jpg"
                    page.save(temp_image_path)

                    if doc_type == "rc":
                        result = predict_from_file(temp_image_path)
                    elif doc_type == "cin":
                        result = extract_fields_cin(temp_image_path)
                    elif doc_type == "rib":
                        result = extract_fields_rib(temp_image_path)
                    else:
                        result = {}

                    # ✅ Fusion intelligente des résultats
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if key not in merged_result:
                                merged_result[key] = value
                            elif merged_result[key] in ["", [], "Non trouvé", None] and value not in ["", [], "Non trouvé", None]:
                                merged_result[key] = value

                    os.remove(temp_image_path)

            else:
                image = Image.open(temp_path).convert("RGB")
                doc_type = classify_document(image)

                if doc_type == "rc":
                    result = predict_from_file(temp_path)
                elif doc_type == "cin":
                    result = extract_fields_cin(temp_path)
                elif doc_type == "rib":
                    result = extract_fields_rib(temp_path)
                else:
                    result = {"error": "Document non reconnu"}

                # ✅ Fusion intelligente des résultats
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in merged_result:
                            merged_result[key] = value
                        elif merged_result[key] in ["", [], "Non trouvé", None] and value not in ["", [], "Non trouvé", None]:
                            merged_result[key] = value

        except Exception as e:
            merged_result["error"] = str(e)

        os.remove(temp_path)

    return JSONResponse(content=merged_result)