import os
import re
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image

ocr = PaddleOCR(lang='fr')

def clean_ocr_text(text):
    return re.sub(r'\\s+', '', text)

def extract_rib_from_text(text):
    text = clean_ocr_text(text)
    number_sequences = re.findall(r'\\d+', text)

    iban_match = re.search(r'MA64\\d{24}', text)
    iban = iban_match.group(0) if iban_match else None
    rib_from_iban = iban[4:] if iban else None

    rib_candidates = []
    i = 0
    while i < len(number_sequences):
        num = number_sequences[i]
        if len(num) == 24 and not num.startswith("MA"):
            rib_candidates.append(num)
        elif len(num) == 48 and num[:24] == num[24:]:
            rib_candidates.append(num[:24])
        elif len(num) == 22 and i + 1 < len(number_sequences):
            next_num = number_sequences[i + 1]
            if len(next_num) == 2:
                combined = num + next_num
                if len(combined) == 24:
                    rib_candidates.append(combined)
                    i += 1
        i += 1

    for rib in rib_candidates:
        if rib_from_iban:
            if rib == rib_from_iban:
                return rib
        else:
            return rib

    if rib_from_iban:
        return rib_from_iban

    return None

def extract_rib_from_image(image_path):
    results = ocr.ocr(image_path, cls=True)
    extracted_text = "\\n".join([line[1][0] for line in results[0] if line[1]])
    return extract_rib_from_text(extracted_text)

def extract_fields_rib(file_path):
    ext = os.path.splitext(file_path.lower())[-1]
    if ext == '.pdf':
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            temp_image_path = f"temp_rib_page_{i+1}.jpg"
            image.save(temp_image_path, "JPEG")
            rib = extract_rib_from_image(temp_image_path)
            os.remove(temp_image_path)
            if rib:
                return {"rib": rib}
        return {"rib": "Non trouvé"}
    elif ext in ['.jpg', '.jpeg', '.png']:
        rib = extract_rib_from_image(file_path)
        return {"rib": rib} if rib else {"rib": "Non trouvé"}
    else:
        return {"rib": "Format non supporté"}