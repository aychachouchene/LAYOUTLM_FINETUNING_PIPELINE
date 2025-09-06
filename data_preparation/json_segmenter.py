import json

MAX_LEN = 512
INPUT_FILE = "layoutlm_formatted_input.json"
OUTPUT_FILE = "layoutlm_segmented_input.json"

def split_example(example):
    segments = []
    total_len = len(example["input_ids"])

    for start in range(0, total_len, MAX_LEN):
        end = min(start + MAX_LEN, total_len)

        segment = {
            "input_ids": example["input_ids"][start:end],
            "bbox": example["bbox"][start:end],
            "attention_mask": example["attention_mask"][start:end],
            "labels": example["labels"][start:end]
        }

        segments.append(segment)
    
    return segments

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_segments = []
    for example in data:
        all_segments.extend(split_example(example))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(all_segments, f_out, indent=2)

    print(f"✅ Fichier segmenté sauvegardé dans : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
