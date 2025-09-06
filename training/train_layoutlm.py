import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import os

# -----------------------------
# PARAMÈTRES
# -----------------------------
label2id = {
    "B-compte": 0, "I-compte": 1, "B-solde_an": 2, "I-solde_an": 3, "B-debit": 4, "I-debit": 5,
    "B-credit": 6, "I-credit": 7, "B-solde": 8, "I-solde": 9, "B-intitulé": 10, "I-intitulé": 11,
    "B-société": 12, "I-société": 13, "B-adresses": 14, "I-adresses": 15, "B-Date_edition": 16,
    "I-Date_edition": 17, "B-Periode": 18, "I-Periode": 19, "B-Date_cloture": 20, "I-Date_cloture": 21,
    "B-Dossier": 22, "I-Dossier": 23, "B-compte_header": 24, "I-compte_header": 25,
    "B-debit_header": 26, "I-debit_header": 27, "B-credit_header": 28, "I-credit_header": 29,
    "B-solde_header": 30, "I-solde_header": 31, "B-intitule_header": 32, "I-intitule_header": 33,
    "B-solde_an_header": 34, "I-solde_an_header": 35, "O": 36
}
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

# -----------------------------
# DATASET
# -----------------------------
class LayoutLMDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "bbox": torch.tensor(example["bbox"], dtype=torch.long),
            "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long)
        }

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "bbox": torch.stack([item["bbox"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# -----------------------------
# ENTRAÎNEMENT
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Loss moyenne sur epoch: {avg_loss:.4f}")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    data_path = input("📂 Chemin du fichier JSON d'entraînement : ").strip()
    while not os.path.isfile(data_path):
        print("❌ Fichier introuvable. Réessaie.")
        data_path = input("📂 Chemin du fichier JSON d'entraînement : ").strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Utilisation de : {device}")

    model = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    dataset = LayoutLMDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(2):
        print(f"\n📘 Epoch {epoch+1}/2")
        train_one_epoch(model, dataloader, optimizer, device)

    model.save_pretrained("mon_model_layoutlm")
    print("✅ Modèle sauvegardé dans le dossier 'mon_model_layoutlm'")
