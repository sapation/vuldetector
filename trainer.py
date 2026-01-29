import os
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from vul_detector import VulDetector
from vul_trainer import VulTrainerManual

# -------------------------
# 0) Load JSONL -> HF Dataset
# -------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

DATASET_DIR = os.path.join(os.getcwd(), "data")
assert os.path.isdir(DATASET_DIR), f"Dataset dir not found: {DATASET_DIR}"

train_dataset = Dataset.from_list(load_jsonl(os.path.join(DATASET_DIR, "train.jsonl")))
valid_dataset = Dataset.from_list(load_jsonl(os.path.join(DATASET_DIR, "val.jsonl")))
test_dataset = Dataset.from_list(load_jsonl(os.path.join(DATASET_DIR, "test.jsonl")))

print("Loaded:", len(train_dataset), len(valid_dataset), len(test_dataset))
print("Example row keys:", train_dataset.column_names)
print("Example:", {k: train_dataset[0][k] for k in ["id", "project", "target", "answer_text"]})


# -------------------------
# 2) Tokenizer + Map
# -------------------------
model_name = "microsoft/unixcoder-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

MAX_LEN = 256  # start 256; try 384 if needed (1024 will likely OOM)

def tokenize_batch(batch):
    texts = [c.strip() for c in batch["func_clean"]]

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        add_special_tokens=True,
    )

    # Labels: prefer numeric target
    if "target" in batch:
        enc["labels"] = [int(x) for x in batch["target"]]
    else:
        enc["labels"] = [
            1 if a.strip().lower() == "vulnerable" else 0
            for a in batch["answer_text"]
        ]
    return enc

train_tok = train_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=train_dataset.column_names,
)
valid_tok = valid_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=valid_dataset.column_names,
)
test_tok = test_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=test_dataset.column_names,
)

# -------------------------
# 3) DataLoaders + class weights
# -------------------------
columns = ["input_ids", "attention_mask", "labels"]
train_tok.set_format(type="torch", columns=columns)
valid_tok.set_format(type="torch", columns=columns)
test_tok.set_format(type="torch", columns=columns)

train_loader = DataLoader(train_tok, batch_size=16, shuffle=True)
val_loader = DataLoader(valid_tok, batch_size=32)
test_loader = DataLoader(test_tok, batch_size=32)

label_counts = Counter(train_dataset["target"])
total = sum(label_counts.values())
class_weights = [total / (2 * label_counts[i]) for i in range(2)]
print("Class weights:", class_weights)
print("Batches -> train:", len(train_loader), "val:", len(val_loader), "test:", len(test_loader))

# -------------------------
# 4) Train vulnerability detector
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VulDetector(model_name=model_name, num_labels=2)
trainer = VulTrainerManual(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    class_weights=class_weights,
    learning_rate=2e-5,
    num_epochs=5,
    loss_type="focal",
    focal_gamma=2.0,
)

trainer.train()

# -------------------------
# 5) Evaluate on validation/test splits
# -------------------------
def evaluate_loader(loader):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = trainer.criterion(logits, labels_batch)
            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(labels_batch.cpu().tolist())
    metrics = trainer.compute_metrics(np.array(preds), np.array(labels))
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics

best_ckpts = sorted(Path(".").glob("best_model_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
if best_ckpts:
    best_ckpt = best_ckpts[-1]
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.to(device)
    print(f"Loaded best checkpoint: {best_ckpt}")
else:
    print("No saved checkpoints found; evaluating current model state.")

val_metrics = evaluate_loader(val_loader)
print("Validation metrics:", val_metrics)

test_metrics = evaluate_loader(test_loader)
print("Test metrics:", test_metrics)


# -------------------------
# 6) Inference function
# -------------------------
def predict_vulnerability(
    code_text: str,
    model_path: str = None,
    model_instance=None,
    tokenizer_instance=None,
    device_instance=None,
    return_probabilities: bool = False
):
    """
    Predict whether a code snippet is vulnerable or safe.
    
    Args:
        code_text: The source code function to analyze
        model_path: Path to saved checkpoint (e.g., 'best_model_epoch_1.pt')
        model_instance: Pre-loaded model (if None, will use global 'model')
        tokenizer_instance: Pre-loaded tokenizer (if None, will use global 'tokenizer')
        device_instance: Device to run on (if None, will use global 'device')
        return_probabilities: If True, return class probabilities instead of label
    
    Returns:
        If return_probabilities=False: "safe" or "vulnerable"
        If return_probabilities=True: dict with probabilities for each class
    """
    # Use provided instances or fall back to globals
    model_to_use = model_instance if model_instance is not None else model
    tokenizer_to_use = tokenizer_instance if tokenizer_instance is not None else tokenizer
    device_to_use = device_instance if device_instance is not None else device
    
    # Load checkpoint if provided
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device_to_use)
        model_to_use.load_state_dict(checkpoint)
        model_to_use.to(device_to_use)
        print(f"Loaded checkpoint: {model_path}")
    
    # Tokenize input
    inputs = tokenizer_to_use(
        code_text.strip(),
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device_to_use) for k, v in inputs.items()}
    
    # Run inference
    model_to_use.eval()
    with torch.no_grad():
        outputs = model_to_use(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    
    if return_probabilities:
        return {
            "safe": probs[0][0].item(),
            "vulnerable": probs[0][1].item(),
            "predicted_class": predicted_class
        }
    else:
        return "vulnerable" if predicted_class == 1 else "safe"


def predict_batch(
    code_texts: list,
    model_path: str = None,
    batch_size: int = 32,
    return_probabilities: bool = False
):
    """
    Predict vulnerability for a batch of code snippets.
    
    Args:
        code_texts: List of source code functions to analyze
        model_path: Path to saved checkpoint (optional)
        batch_size: Batch size for inference
        return_probabilities: If True, return probabilities instead of labels
    
    Returns:
        List of predictions (either labels or probability dicts)
    """
    # Load checkpoint if provided
    model_to_use = model
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model_to_use.load_state_dict(checkpoint)
        model_to_use.to(device)
        print(f"Loaded checkpoint: {model_path}")
    
    model_to_use.eval()
    predictions = []
    
    # Process in batches
    for i in range(0, len(code_texts), batch_size):
        batch_texts = code_texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            [text.strip() for text in batch_texts],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model_to_use(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)
        
        # Collect results
        for j in range(len(batch_texts)):
            if return_probabilities:
                predictions.append({
                    "safe": probs[j][0].item(),
                    "vulnerable": probs[j][1].item(),
                    "predicted_class": predicted_classes[j].item()
                })
            else:
                pred_class = predicted_classes[j].item()
                predictions.append("vulnerable" if pred_class == 1 else "safe")
    
    return predictions


# Example usage:
if __name__ == "__main__":
    # Single prediction example
    sample_code = """
    void unsafe_copy(char *dest, char *src) {
        strcpy(dest, src);  // No bounds checking
    }
    """
    
    # Using the best saved checkpoint
    result = predict_vulnerability(
        sample_code,
        model_path="best_model_epoch_1.pt",
        return_probabilities=True
    )
    print(f"\nSingle prediction: {result}")
    
    # Batch prediction example
    sample_codes = [
        "int safe_add(int a, int b) { return a + b; }",
        "void unsafe(char *buf) { gets(buf); }",
    ]
    
    results = predict_batch(sample_codes, model_path="best_model_epoch_1.pt")
    print(f"\nBatch predictions: {results}")