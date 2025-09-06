import json
import torch
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_DIR = "mon_model_layoutlm"
TEST_INPUT_FILE = "layoutlm_test_input.json"
OUTPUT_FILE = "layoutlm_predictions_output.json"

label_map = {
    0: "B-compte", 1: "I-compte", 2: "B-solde_an", 3: "I-solde_an", 4: "B-debit", 5: "I-debit",
    6: "B-credit", 7: "I-credit", 8: "B-solde", 9: "I-solde", 10: "B-intitulé", 11: "I-intitulé",
    12: "B-société", 13: "I-société", 14: "B-adresses", 15: "I-adresses", 16: "B-Date_edition",
    17: "I-Date_edition", 18: "B-Periode", 19: "I-Periode", 20: "B-Date_cloture", 21: "I-Date_cloture",
    22: "B-Dossier", 23: "I-Dossier", 24: "B-compte_header", 25: "I-compte_header",
    26: "B-debit_header", 27: "I-debit_header", 28: "B-credit_header", 29: "I-credit_header",
    30: "B-solde_header", 31: "I-solde_header", 32: "B-intitule_header", 33: "I-intitule_header",
    34: "B-solde_an_header", 35: "I-solde_an_header", 36: "O"
}

# -----------------------------
# CHARGEMENT MODELE + TOKENIZER
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoutLMForTokenClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

# -----------------------------
# TEST / PREDICTION
# -----------------------------
with open(TEST_INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

all_predictions = []

for sample in data:
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    bbox = torch.tensor(sample["bbox"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    words = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    boxes = sample["bbox"]

    for word, box, pred in zip(words, boxes, predictions):
        all_predictions.append({
            "word": word,
            "bbox": box,
            "predicted_label": label_map[pred]
        })

# -----------------------------
# ENREGISTREMENT DES PRÉDICTIONS
# -----------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_predictions, f, indent=2, ensure_ascii=False)

print(f"✅ Prédictions sauvegardées dans : {OUTPUT_FILE}")
