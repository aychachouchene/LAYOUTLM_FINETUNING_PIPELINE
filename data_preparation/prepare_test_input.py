import json
from transformers import LayoutLMTokenizer

def convert_ls_to_layoutlm_test_format(ls_path, output_path):
    # Charger le tokenizer LayoutLM
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    with open(ls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words = []
    bboxes = []

    # Parcours des tokens et boîtes issus de Label Studio
    for item in data:
        results = item['predictions'][0]['result']
        for i in range(0, len(results), 2):
            rect = results[i]
            text = results[i + 1]

            # Vérification des types attendus
            if rect['type'] != 'rectangle' or text['type'] != 'textarea':
                continue

            value = rect['value']
            word = text['value']['text'][0] if text['value']['text'] else ""

            # Dimensions originales de l'image (pixel)
            width = value["original_width"]
            height = value["original_height"]

            # Conversion coordonnées % → pixels
            x0 = int((value['x'] / 100) * width)
            y0 = int((value['y'] / 100) * height)
            x1 = int(((value['x'] + value['width']) / 100) * width)
            y1 = int(((value['y'] + value['height']) / 100) * height)

            # Normalisation bbox entre 0 et 1000 pour LayoutLM
            norm_bbox = [
                int(1000 * x0 / width),
                int(1000 * y0 / height),
                int(1000 * x1 / width),
                int(1000 * y1 / height)
            ]

            words.append(word)
            bboxes.append(norm_bbox)

    # Tokenisation avec sous-tokens et duplication bbox par sous-token
    tokens = []
    final_bboxes = []

    for word, box in zip(words, bboxes):
        word_tokens = tokenizer.tokenize(word)
        token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

        tokens.extend(token_ids)
        final_bboxes.extend([box] * len(token_ids))

    # Tronquer si trop long (>512 tokens)
    if len(tokens) > 512:
        tokens = tokens[:512]
        final_bboxes = final_bboxes[:512]

    # Padding si trop court
    while len(tokens) < 512:
        tokens.append(tokenizer.pad_token_id)
        final_bboxes.append([0, 0, 0, 0])

    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in tokens]

    # Labels factices tous à 0 ("O")
    labels = [0] * 512

    processed_sample = {
        "input_ids": tokens,
        "attention_mask": attention_mask,
        "bbox": final_bboxes,
        "labels": labels
    }

    # Sauvegarde résultante JSON (liste pour compatibilité avec DataLoader)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump([processed_sample], f_out, indent=2, ensure_ascii=False)

    print(f"✅ Fichier de test LayoutLM généré : {output_path}")


if __name__ == "__main__":
    # Remplace par ton chemin source Label Studio JSON brut
    input_ls_path = "BG_QUADRA_full_tasks - Copie.json"
    output_layoutlm_test_path = "layoutlm_test_input.json"

    convert_ls_to_layoutlm_test_format(input_ls_path, output_layoutlm_test_path)
