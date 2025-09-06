import json

TOL = 0.5  # Tolérance de position (en %)

def are_boxes_close(box1, box2, tol):
    for key in ['x', 'y', 'width', 'height']:
        if abs(box1[key] - box2[key]) > tol:
            return False
    return True

def fusionner_label_token_boxes(ls_data, ocr_tokens, tol=0.5):
    final_boxes = []

    # Extraire tous les rectangles OCR avec leur texte
    ocr_rects = []
    for pred in ocr_tokens[0]["predictions"]:
        results = pred["result"]
        for i in range(0, len(results), 2):
            rect = results[i]
            text = results[i+1]
            if rect["type"] == "rectangle" and text["type"] == "textarea":
                box = rect["value"]
                txt = text["value"]["text"][0] if text["value"]["text"] else ""
                ocr_rects.append({
                    "x": box["x"],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"],
                    "text": txt
                })

    for task in ls_data:
        for box in task["bbox"]:
            # Chercher le label éventuel
            label = box.get("rectanglelabels", ["O"])
            label = label[0] if isinstance(label, list) and label else "O"

            # Chercher le texte OCR correspondant
            matched_text = ""
            for ocr in ocr_rects:
                if (
                    abs(ocr["x"] - box["x"]) < tol and
                    abs(ocr["y"] - box["y"]) < tol and
                    abs(ocr["width"] - box["width"]) < tol and
                    abs(ocr["height"] - box["height"]) < tol
                ):
                    matched_text = ocr["text"]
                    break

            final_boxes.append({
                "text": matched_text,
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "label": label
            })

    return final_boxes


def main():
    LS_FILE = "testLLM.json"
    OCR_FILE = "BALANCE_QUADRA_AUTRE_FORMAT_full_tasks - Copie.json"
    OUTPUT_FILE = "BALANCE_QUADRA_AUTRE_FORMAT_full_tasks_enriched.json"

    with open(LS_FILE, "r", encoding="utf-8") as f1:
        ls_boxes = json.load(f1)

    with open(OCR_FILE, "r", encoding="utf-8") as f2:
        ocr_data = json.load(f2)

    final_boxes = fusionner_label_token_boxes(ls_boxes, ocr_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(final_boxes, f_out, indent=2, ensure_ascii=False)

    print("✅ Fusion terminée. Résultat sauvegardé dans", OUTPUT_FILE)


if __name__ == "__main__":
    main()
