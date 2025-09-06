import json
from transformers import LayoutLMTokenizer
from tqdm import tqdm

# Mise à jour complète du mapping label vers ID
label2id = {
    "B-compte": 0,
    "I-compte": 1,
    "B-solde_an": 2,
    "I-solde_an": 3,
    "B-debit": 4,
    "I-debit": 5,
    "B-credit": 6,
    "I-credit": 7,
    "B-solde": 8,
    "I-solde": 9,
    "B-intitulé": 10,
    "I-intitulé": 11,
    "B-société": 12,
    "I-société": 13,
    "B-adresses": 14,
    "I-adresses": 15,
    "B-Date_edition": 16,
    "I-Date_edition": 17,
    "B-Periode": 18,
    "I-Periode": 19,
    "B-Date_cloture": 20,
    "I-Date_cloture": 21,
    "B-Dossier": 22,
    "I-Dossier": 23,
    "B-compte_header": 24,
    "I-compte_header": 25,
    "B-debit_header": 26,
    "I-debit_header": 27,
    "B-credit_header": 28,
    "I-credit_header": 29,
    "B-solde_header": 30,
    "I-solde_header": 31,
    "B-intitule_header": 32,
    "I-intitule_header": 33,
    "B-solde_an_header": 34,
    "I-solde_an_header": 35,
    "O": 36  # label Outside ou non annoté
}

def normalize_bbox(x, y, width, height):
    """
    Normalise bbox sur 0-1000.
    x, y, width, height sont en pourcentage (0-100)
    """
    x0 = int(x * 10)
    y0 = int(y * 10)
    x1 = int((x + width) * 10)
    y1 = int((y + height) * 10)

    # S'assurer que les coordonnées restent dans [0, 1000]
    x0 = max(0, min(1000, x0))
    y0 = max(0, min(1000, y0))
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))

    return [x0, y0, x1, y1]

def process_json_for_layoutlm(input_json_path, output_json_path):
    # Chargement du fichier JSON source
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    input_ids = []
    bboxes = []
    attention_mask = []
    labels = []

    for entry in tqdm(data, desc="Processing tokens"):
        text = entry['text']
        label_str = entry['label']
        # Si le label n'est pas dans label2id, on considère "O"
        label_id = label2id.get(label_str, label2id["O"])
        bbox_norm = normalize_bbox(entry['x'], entry['y'], entry['width'], entry['height'])

        # Tokenisation en sous-tokens
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        for tid in token_ids:
            input_ids.append(tid)
            bboxes.append(bbox_norm)
            attention_mask.append(1)
            labels.append(label_id)

    processed_data = {
        "input_ids": input_ids,
        "bbox": bboxes,
        "attention_mask": attention_mask,
        "labels": labels
    }

    # Sauvegarde dans un fichier JSON
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(processed_data, f_out, ensure_ascii=False, indent=2)

    print(f"Fichier prêt pour LayoutLM sauvegardé sous : {output_json_path}")

if __name__ == "__main__":
    input_json_path = "BALANCE_QUADRA_AUTRE_FORMAT_full_tasks_enriched.json"  # mets ici le chemin vers ton JSON d’entrée
    output_json_path = "layoutlm_formatted_input.json"
    process_json_for_layoutlm(input_json_path, output_json_path)
