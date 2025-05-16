import pytesseract
from pytesseract import Output
from PIL import Image
from pdf2image import convert_from_path
import json
import re
import numpy as np
import os
import unicodedata
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# ====== Chargement du modèle LayoutLMv3
processor = LayoutLMv3Processor.from_pretrained("ghitadrh/layoutlmv3-kyc", apply_ocr=False)
processor.tokenizer.add_tokens(["<ARABIC>"])
model = LayoutLMv3ForTokenClassification.from_pretrained("ghitadrh/layoutlmv3-kyc")
model.eval()
id2label = model.config.id2label

# ====== OCR utils version locale (fiable)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (a2 - a1) * (b2 - b1)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def group_lines(words, bboxes, dynamic_threshold_factor=0.7):
    valid_data = [(word, bbox) for word, bbox in zip(words, bboxes) if len(bbox) == 4]
    if not valid_data:
        return [], []

    sorted_data = sorted(valid_data, key=lambda x: x[1][1])
    heights = [bbox[3] - bbox[1] for _, bbox in valid_data]
    median_height = np.median(heights) if heights else 10
    y_threshold = median_height * dynamic_threshold_factor

    lines = []
    current_line = []
    current_y = sorted_data[0][1][1]

    for word, bbox in sorted_data:
        if abs(bbox[1] - current_y) <= y_threshold:
            current_line.append((word, bbox))
        else:
            current_line.sort(key=lambda x: x[1][0])
            lines.append(current_line)
            current_line = [(word, bbox)]
            current_y = bbox[1]

    if current_line:
        current_line.sort(key=lambda x: x[1][0])
        lines.append(current_line)

    words_sorted, bboxes_sorted = [], []
    for line in lines:
        for word, bbox in line:
            words_sorted.append(word)
            bboxes_sorted.append(bbox)

    return words_sorted, bboxes_sorted

def extract_ocr(image):
    data_mixed = pytesseract.image_to_data(image, output_type=Output.DICT, lang='ara+fra')
    data_fra = pytesseract.image_to_data(image, output_type=Output.DICT, lang='fra')

    words, bboxes = [], []

    for i in range(len(data_mixed["text"])):
        text = data_mixed["text"][i].strip()
        if not text:
            continue

        x, y, w, h = data_mixed["left"][i], data_mixed["top"][i], data_mixed["width"][i], data_mixed["height"][i]
        bbox = [x, y, x + w, y + h]

        if '\u200e' in text:
            best_match = ""
            max_iou = 0

            for j in range(len(data_fra["text"])):
                text_fra = data_fra["text"][j].strip()
                if not text_fra:
                    continue

                xf, yf, wf, hf = data_fra["left"][j], data_fra["top"][j], data_fra["width"][j], data_fra["height"][j]
                bbox_fra = [xf, yf, xf + wf, yf + hf]
                score = iou(bbox, bbox_fra)
                if score > max_iou:
                    max_iou = score
                    best_match = text_fra

            words.append(best_match if best_match else text.replace('\u200e', '').replace('\u200f', ''))
        else:
            words.append(text)

        bboxes.append(bbox)

    return group_lines(words, bboxes), image.size


def is_arabic(word):
    pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(pattern.search(str(word)))

def normalize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def preprocess_words(words):
    return ["<ARABIC>" if is_arabic(w) else normalize_text(w) for w in words]

def normalize_bbox(bbox, image_size):
    width, height = image_size
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

def clean_cin(cin):
    return re.sub(r'[^a-zA-Z0-9]', '', cin).upper()

def main_label_type(label):
    return label.replace("B-", "").replace("I-", "").replace("E-", "").replace("S-", "")

def apply_strict_sequence(results):
    expected_sequence = ["ICE", "RC", "DÉNOMINATION", "ACTIVITÉ", "SIÈGE SOCIAL", "FORME JURIDIQUE"]
    last_index = len(expected_sequence) - 1
    sequence_pointer = 0
    current_entity = []
    current_label = None
    corrected_results = []
    seen_entities = set()
    strict_mode = True
    for item in results:
        raw_label = item["label"]
        label_type = main_label_type(raw_label)
        if label_type == "O":
            if current_entity:
                corrected_results.extend(current_entity)
                current_entity = []
                current_label = None
            corrected_results.append(item)
            continue
        if strict_mode and sequence_pointer > last_index:
            strict_mode = False
        if not strict_mode:
            if current_entity and label_type != current_label:
                corrected_results.extend(current_entity)
                current_entity = []
            if label_type == current_label:
                current_entity.append(item)
            else:
                current_entity = [item]
                current_label = label_type
            continue
        while sequence_pointer < len(expected_sequence):
            expected = expected_sequence[sequence_pointer]
            if expected == "ICE" and "RC" in seen_entities:
                sequence_pointer += 1
            else:
                break
        expected_now = expected_sequence[sequence_pointer] if sequence_pointer < len(expected_sequence) else None
        if expected_now == "ICE" and label_type == "RC":
            expected_now = "RC"
            seen_entities.add("ICE")
            sequence_pointer += 1
        if label_type == expected_now:
            current_entity.append(item)
            current_label = label_type
            seen_entities.add(label_type)
            if label_type == expected_sequence[last_index]:
                strict_mode = False
                sequence_pointer += 1
        elif label_type in expected_sequence and label_type != expected_now:
            if current_entity:
                corrected_results.extend(current_entity)
                current_entity = []
            sequence_pointer += 1
            current_entity = [item]
            current_label = label_type
            seen_entities.add(label_type)
        elif label_type == current_label:
            current_entity.append(item)
        else:
            if current_entity:
                corrected_results.extend(current_entity)
                current_entity = []
            item["label"] = "O"
            corrected_results.append(item)
            current_label = None
    if current_entity:
        corrected_results.extend(current_entity)
    return corrected_results


def clean_text_entity(words):
    text = " ".join(words).strip()
    text = re.sub(r"^[^a-zA-Z0-9(<]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9)>]+$", "", text)
    return text.strip()

def get_forme_juridique_abbreviation(forme_words):
    mapping = {
        "societe a responsabilite limitee a associe unique": "SARL AU",
        "societe a responsabilite limitee": "SARL",
        "societe anonyme": "SA",
        "societe par actions simplifiee unipersonnelle": "SASU",
        "societe par actions simplifiee": "SAS",
        "societe en nom collectif": "SNC",
        "societe en commandite simple": "SCS",
        "societe en commandite par actions": "SCA",
        "entreprise individuelle": "EI",
        "cooperative": "Coopérative",
        "groupement dinteret economique": "GIE",
        "societe civile": "Société Civile"
    }
    texte = normalize_text(" ".join(forme_words).lower())
    texte = re.sub(r"[^\w\s]", "", texte)
    for full, abbr in mapping.items():
        if full in texte:
            return abbr
    return " ".join(forme_words)

def filter_false_positives_box(results):
    for i, item in enumerate(results):
        word, label = item["word"], item["label"]
        if label in ["B-CIN", "S-CIN"] and re.search(r"C[\W_]*[1I][\W_]*N", word, re.IGNORECASE):
            results[i]["label"] = "O"
        if label.endswith("DIRIGEANT"):
            window = results[i:i+5]
            for w in window:
                if re.search(r'\b(RC|R\\.C\\.|RCS)\b', w["word"], re.IGNORECASE):
                    results[i]["label"] = "O"
                    break
    return results

def correct_bad_societe_predictions_after_text(results):
    forme_index = -1
    for i, item in enumerate(results):
        if "forme" in item["word"].lower() and i + 1 < len(results):
            if "juridique" in results[i + 1]["word"].lower():
                forme_index = i + 1
    if forme_index == -1:
        return results
    for i in range(forme_index + 1, min(len(results), forme_index + 11)):
        if re.fullmatch(r"\d{5,}", results[i]["word"]):
            results[i]["word"] = "SOCIETE"
            results[i]["label"] = "S-FORME JURIDIQUE"
            break
    return results

def structure_results_box(results):
    structured = {"RC": [], "ICE": [], "DÉNOMINATION": [], "ACTIVITÉ": [], "FORME JURIDIQUE": [], "SIÈGE SOCIAL": [], "DIRIGEANTS": []}
    current_dirigeant, current_cin = [], None
    for item in results:
        label, word = item["label"], item["word"]
        if "RC" in label:
            cleaned = re.sub(r"[^\d]", "", word)
            if cleaned: structured["RC"].append(cleaned)
        elif "ICE" in label:
            matches = re.findall(r"\d{6,}", word)
            if matches:
                cleaned = max(matches, key=len)
                structured["ICE"].append(cleaned)
        elif "DÉNOMINATION" in label:
            structured["DÉNOMINATION"].append(word)
        elif "ACTIVITÉ" in label:
            structured["ACTIVITÉ"].append(word)
        elif "FORME JURIDIQUE" in label:
            structured["FORME JURIDIQUE"].append(word)
        elif "SIÈGE SOCIAL" in label:
            structured["SIÈGE SOCIAL"].append(word)
        elif "-DIRIGEANT" in label:
            if current_dirigeant and current_cin:
                structured["DIRIGEANTS"].append({"nom": " ".join(current_dirigeant), "cin": current_cin})
                current_dirigeant, current_cin = [], None
            current_dirigeant.append(word)
        elif "-CIN" in label and current_dirigeant:
            if current_cin:
                current_cin += clean_cin(word)
            else:
                current_cin = clean_cin(word)
    if current_dirigeant:
        structured["DIRIGEANTS"].append({"nom": " ".join(current_dirigeant), "cin": current_cin if current_cin else "Non trouvé"})
    return structured

def predict_from_file(input_path):
    print("🔍 Début prédiction fichier :", input_path)
    all_words, all_bboxes, all_sizes, images = [], [], [], []
    if input_path.lower().endswith(".pdf"):
        pages = convert_from_path(input_path)
        for i, page in enumerate(pages):
            (words, bboxes), size = extract_ocr(page)
            print(f"📄 Page {i+1} : {len(words)} mots détectés")
            all_words.append(words)
            all_bboxes.append(bboxes)
            all_sizes.append(size)
            images.append(page)
    else:
        image = Image.open(input_path).convert("RGB")
        (words, bboxes), size = extract_ocr(image)
        print("🖼️ Image :", len(words), "mots détectés")
        all_words.append(words)
        all_bboxes.append(bboxes)
        all_sizes.append(size)
        images.append(image)

    results = []
    for idx, (words, bboxes, image_size) in enumerate(zip(all_words, all_bboxes, all_sizes)):
        print(f"🔁 Traitement page {idx+1} : {len(words)} mots")
        if idx == 2:
            continue
        processed_words = preprocess_words(words)
        normalized_bboxes = [normalize_bbox(b, image_size) for b in bboxes]
        encoding = processor(
            images[idx], processed_words, boxes=normalized_bboxes,
            truncation=True, padding="max_length", max_length=512,
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, return_tensors="pt"
        )
        encoding.pop("offset_mapping", None)
        encoding.pop("overflow_to_sample_mapping", None)
        encoding["pixel_values"] = torch.stack([p for p in encoding["pixel_values"]])
        with torch.no_grad():
            outputs = model(**encoding)
        logits = outputs.logits
        predictions = logits.argmax(-1).cpu().numpy()
        token_boxes = encoding["bbox"].cpu().numpy()
        input_ids = encoding["input_ids"].cpu().numpy()
        if token_boxes.ndim == 2:
            token_boxes = [token_boxes]
            predictions = [predictions]
            input_ids = [input_ids]
        box_token_dict = {}
        for i in range(len(token_boxes)):
            initial_j = 0 if i == 0 else 129
            for j in range(initial_j, len(token_boxes[i])):
                box = token_boxes[i][j]
                pred = predictions[i][j]
                token_id = input_ids[i][j]
                if list(box) == [0, 0, 0, 0]: continue
                box_tuple = tuple(box.tolist())
                token = processor.tokenizer.decode([token_id]).strip()
                if not token: continue
                if box_tuple not in box_token_dict:
                    box_token_dict[box_tuple] = {"tokens": [token], "labels": [pred]}
                else:
                    box_token_dict[box_tuple]["tokens"].append(token)
                    box_token_dict[box_tuple]["labels"].append(pred)
        for box, info in box_token_dict.items():
            tokens = info["tokens"]
            labels = info["labels"]
            non_o_labels = [l for l in labels if id2label[l] != "O"]
            label_id = max(set(non_o_labels), key=non_o_labels.count) if non_o_labels else max(set(labels), key=labels.count)
            word = "".join(tokens).replace("▁", " ").strip()
            label_final = id2label[label_id]
            print(f"🔎 Mot: '{word}' | Label: {label_final}")
            results.append({"bbox": box, "word": word, "label": label_final})

    print("🔧 Application séquence stricte")
    results = apply_strict_sequence(results)
    print("🧹 Nettoyage faux positifs")
    results = filter_false_positives_box(results)
    print("🔧 Correction formes juridiques")
    results = correct_bad_societe_predictions_after_text(results)
    print("📦 Structuration des résultats")
    results = structure_results_box(results)
    return {
        "RC": clean_text_entity(results["RC"]),
        "ICE": clean_text_entity(results["ICE"]),
        "DÉNOMINATION": clean_text_entity(results["DÉNOMINATION"]),
        "ACTIVITÉ": clean_text_entity(results["ACTIVITÉ"]),
        "FORME JURIDIQUE": get_forme_juridique_abbreviation(results["FORME JURIDIQUE"]),
        "SIÈGE SOCIAL": clean_text_entity(results["SIÈGE SOCIAL"]),
        "DIRIGEANTS": results["DIRIGEANTS"]
    }