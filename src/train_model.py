"""
PyTorch MLP for question difficulty classification.
Architecture: 384-dim sentence embedding → 128 → 64 → 3 classes (easy/medium/hard)

Fixes vs previous version
──────────────────────────
• Loads pre-split train/val/test files — no augmented-twin leakage
• Val set used exclusively for scheduler + early stopping
• Test set touched only once, at the very end
• Label smoothing replaces plain CrossEntropyLoss
• Smaller architecture (right-sized for dataset)
• weights_only=True on torch.load (PyTorch ≥ 2.0 compatibility)
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ─── Config ───────────────────────────────────────────────────────────────────

LABEL_MAP    = {"easy": 0, "medium": 1, "hard": 2}
INV_LABEL    = {0: "easy", 1: "medium", 2: "hard"}
EMBED_MODEL  = "all-MiniLM-L6-v2"
MODEL_PATH   = "models/difficulty_model.pt"
STATS_PATH   = "models/training_stats.json"

HIDDEN       = 128
DROPOUT      = 0.35
LR           = 3e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTH = 0.12
MAX_EPOCHS   = 80
PATIENCE     = 12
BATCH_TRAIN  = 16
BATCH_EVAL   = 32
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class QuestionDataset(Dataset):
    def __init__(self, items: list, embedder: SentenceTransformer):
        texts  = [it["question"]   for it in items]
        labels = [LABEL_MAP[it["difficulty"]] for it in items]
        embs   = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self.X = torch.tensor(embs,   dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── Model ────────────────────────────────────────────────────────────────────

class DifficultyClassifier(nn.Module):
    """
    Deliberately kept small — 384 → 128 → 64 → 3.
    ~60 k parameters for ~160 training samples avoids the overfitting
    that the previous 384→256→128→64→3 architecture produced.
    """
    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LayerNorm(HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN // 2, 3),
        )

    def forward(self, x):
        return self.net(x)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def smooth_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         smoothing: float = LABEL_SMOOTH) -> torch.Tensor:
    """
    Label-smoothed cross-entropy.
    Prevents the model from becoming over-confident on clean/easy labels,
    which is the main cause of artificially perfect val scores.
    """
    n_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    nll    = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    loss   = (1.0 - smoothing) * nll + smoothing * smooth
    return loss.mean()


# ─── Eval helper ──────────────────────────────────────────────────────────────

def _evaluate(model: nn.Module, loader: DataLoader):
    """Returns (avg_loss, accuracy, preds_list, labels_list)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels  = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits  = model(xb)
            total_loss += smooth_cross_entropy(logits, yb).item() * len(yb)
            preds   = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            n       += len(yb)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())

    return total_loss / n, correct / n, all_preds, all_labels


# ─── Training ────────────────────────────────────────────────────────────────

def train(
    train_path: str = "data/train.json",
    val_path:   str = "data/val.json",
    test_path:  str = "data/test.json",
) -> dict:
    """
    Full training loop.

    Split contract
    ──────────────
    train.json  — weight updates only
    val.json    — scheduler stepping + early-stop checkpoint selection
    test.json   — final evaluation, touched exactly once after training ends
    """
    os.makedirs("models", exist_ok=True)
    print(f"Device: {DEVICE}\n")

    # ── Load splits ──────────────────────────────────────────────────────────
    def _load(path):
        with open(path) as f:
            return json.load(f)

    train_data = _load(train_path)
    val_data   = _load(val_path)
    test_data  = _load(test_path)
    print(f"Samples  train:{len(train_data)}  val:{len(val_data)}  test:{len(test_data)}")

    # ── Embedder ─────────────────────────────────────────────────────────────
    print("\nLoading sentence embedder (downloads ~80 MB on first run) …")
    embedder = SentenceTransformer(EMBED_MODEL)

    train_ds = QuestionDataset(train_data, embedder)
    val_ds   = QuestionDataset(val_data,   embedder)
    test_ds  = QuestionDataset(test_data,  embedder)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL)

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model = DifficultyClassifier().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None
    stats_log     = []

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"Training for up to {MAX_EPOCHS} epochs  (early-stop patience={PATIENCE}) …\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            smooth_cross_entropy(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        train_loss, train_acc, _, _ = _evaluate(model, train_loader)
        val_loss,   val_acc,   _, _ = _evaluate(model, val_loader)

        stats_log.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            "train_acc":  round(train_acc,  4),
            "val_acc":    round(val_acc,    4),
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"Train {train_loss:.4f} / {train_acc:.2%} | "
                  f"Val   {val_loss:.4f} / {val_acc:.2%}")

        # early stopping — driven by VAL loss only
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODEL_PATH)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch} "
                      f"(val loss flat for {PATIENCE} epochs)")
                break

    # ── Final evaluation on TEST set (one shot) ───────────────────────────────
    print("\n── Test-set evaluation ──────────────────────────────────────────")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    test_loss, test_acc, preds, labels = _evaluate(model, test_loader)

    pred_names  = [INV_LABEL[p] for p in preds]
    label_names = [INV_LABEL[l] for l in labels]

    print(f"  Loss: {test_loss:.4f}   Accuracy: {test_acc:.2%}\n")
    print(classification_report(label_names, pred_names,
                                  target_names=["easy", "medium", "hard"]))

    cm = confusion_matrix(label_names, pred_names, labels=["easy", "medium", "hard"])
    print("Confusion matrix  (rows = true, cols = predicted):")
    print("              easy  medium  hard")
    for row_label, row in zip(["easy  ", "medium", "hard  "], cm):
        print(f"  {row_label}    {row[0]:4d}  {row[1]:6d}  {row[2]:4d}")

    # ── Persist stats ─────────────────────────────────────────────────────────
    report_dict = classification_report(
        label_names, pred_names,
        target_names=["easy", "medium", "hard"],
        output_dict=True,
    )

    stats = {
        "config": {
            "embed_model":    EMBED_MODEL,
            "architecture":   "384 → 128(LN+ReLU+Drop) → 64(LN+ReLU+Drop) → 3",
            "hidden":         HIDDEN,
            "dropout":        DROPOUT,
            "label_smoothing": LABEL_SMOOTH,
            "lr":             LR,
            "weight_decay":   WEIGHT_DECAY,
            "max_epochs":     MAX_EPOCHS,
            "patience":       PATIENCE,
        },
        "results": {
            "test_accuracy":  round(test_acc, 4),
            "test_loss":      round(test_loss, 4),
            "best_val_loss":  round(best_val_loss, 4),
            "epochs_trained": stats_log[-1]["epoch"],
            "classification_report": report_dict,
            "confusion_matrix": {
                "labels": ["easy", "medium", "hard"],
                "matrix": cm.tolist(),
            },
        },
        "per_epoch": stats_log,
        "split_sizes": {
            "train": len(train_data),
            "val":   len(val_data),
            "test":  len(test_data),
        },
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nModel  → {MODEL_PATH}")
    print(f"Stats  → {STATS_PATH}")
    return stats


# ─── Inference ────────────────────────────────────────────────────────────────

def load_model() -> DifficultyClassifier:
    model = DifficultyClassifier()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def predict_difficulty(
    questions: list[str],
    model: DifficultyClassifier,
    embedder: SentenceTransformer,
) -> list[str]:
    embs = embedder.encode(questions, convert_to_numpy=True)
    X    = torch.tensor(embs, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()
    return [INV_LABEL[int(p)] for p in preds]


def predict_with_confidence(
    questions: list[str],
    model: DifficultyClassifier,
    embedder: SentenceTransformer,
) -> list[dict]:
    embs = embedder.encode(questions, convert_to_numpy=True)
    X    = torch.tensor(embs, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = logits.argmax(dim=1).numpy()
    return [
        {
            "question":    questions[i],
            "difficulty":  INV_LABEL[int(preds[i])],
            "confidence":  round(float(probs[i].max()), 4),
            "prob_easy":   round(float(probs[i][0]), 4),
            "prob_medium": round(float(probs[i][1]), 4),
            "prob_hard":   round(float(probs[i][2]), 4),
        }
        for i in range(len(questions))
    ]


def load_training_stats() -> dict:
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from generate_dataset import generate_dataset
    generate_dataset()
    train()