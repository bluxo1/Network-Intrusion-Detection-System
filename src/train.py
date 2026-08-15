"""
Train the binary and multi-class NIDS models.

Pipeline
--------
1. Load the preprocessed arrays (running ``preprocess.run()`` first if needed).
2. Carve a stratified validation split out of KDDTrain+.
3. Train:
     * the binary Normal-vs-Attack model (BCEWithLogitsLoss, optional pos_weight),
     * the 5-class model (CrossEntropyLoss with inverse-frequency class weights).
4. Apply StepLR decay + early stopping on validation loss.
5. Save each model's ``state_dict`` plus the hyper-parameters needed to rebuild
   it, so inference never has to guess the architecture.

Run with:  ``python -m src.train``
"""

import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from . import preprocess
from .config import CONFIG, abspath
from .dataset import NIDSDataset
from .model import build_models
from .schema import CLASS_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays() -> Dict[str, np.ndarray]:
    """Load processed arrays, regenerating them if the cache is missing."""
    npz_path = os.path.join(abspath(CONFIG["paths"]["processed_dir"]), "dataset.npz")
    if not os.path.exists(npz_path):
        print("[train] processed dataset not found - running preprocessing...")
        preprocess.run()
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Generic training loop with early stopping
# ---------------------------------------------------------------------------
def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tcfg: dict,
    task: str,
) -> nn.Module:
    """Train ``model`` and return it with the best (lowest val-loss) weights."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=tcfg["lr_step_size"], gamma=tcfg["lr_gamma"]
    )

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = tcfg["early_stopping_patience"]
    epochs_no_improve = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        # ---- training ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * xb.size(0)
                if task == "binary":
                    pred = (torch.sigmoid(out) >= 0.5).float()
                    correct += (pred == yb).sum().item()
                else:
                    pred = out.argmax(dim=1)
                    correct += (pred == yb).sum().item()
                total += xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step()

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = "*" if improved else " "
        print(
            f"[{task}] epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} {flag}"
        )

        if epochs_no_improve >= patience:
            print(f"[{task}] early stopping at epoch {epoch} (best val_loss {best_val:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def save_model(model: nn.Module, path: str, input_dim: int, hidden: List[int],
               dropout: List[float], output_dim: int, kind: str) -> None:
    """Save weights + architecture so the model can be rebuilt for inference."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden,
            "dropouts": dropout,
            "output_dim": output_dim,
            "kind": kind,
        },
        path,
    )
    print(f"[train] saved {kind} model -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tcfg = CONFIG["training"]
    mcfg = CONFIG["model"]
    device = resolve_device(tcfg["device"])
    set_seed(tcfg["seed"])
    print(f"[train] device = {device}")

    arrays = load_arrays()
    X_train_full = arrays["X_train"]
    y_train_full = arrays["y_train"]      # multi-class indices
    yb_train_full = arrays["yb_train"]    # binary 0/1
    input_dim = X_train_full.shape[1]
    num_classes = len(CLASS_NAMES)

    # Stratified train/val split (stratify on the 5-class label to keep the rare
    # R2L/U2R classes represented in validation).
    idx = np.arange(len(X_train_full))
    tr_idx, val_idx = train_test_split(
        idx, test_size=tcfg["val_split"], random_state=tcfg["seed"], stratify=y_train_full
    )

    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]
    yb_tr, yb_val = yb_train_full[tr_idx], yb_train_full[val_idx]

    binary_model, multiclass_model = build_models(input_dim, mcfg, num_classes=num_classes)
    binary_model.to(device)
    multiclass_model.to(device)

    # ======================= Binary model =======================
    print("\n=== Training binary classifier (Normal vs Attack) ===")
    bin_train = DataLoader(
        NIDSDataset(X_tr, yb_tr, binary=True), batch_size=tcfg["batch_size"], shuffle=True
    )
    bin_val = DataLoader(
        NIDSDataset(X_val, yb_val, binary=True), batch_size=tcfg["batch_size"], shuffle=False
    )
    # pos_weight balances the (mild) class imbalance to protect attack recall.
    n_pos = float(yb_tr.sum())
    n_neg = float(len(yb_tr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    bin_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    binary_model = train_loop(
        binary_model, bin_train, bin_val, bin_criterion, device, tcfg, task="binary"
    )

    # ======================= Multi-class model =======================
    print("\n=== Training multi-class classifier (5 classes) ===")
    # drop_last=True guards the BatchNorm1d layer in the multi-class model:
    # a trailing batch of size 1 would raise "Expected more than 1 value per
    # channel when training". Dropping it costs <1 batch/epoch of data.
    mc_train = DataLoader(
        NIDSDataset(X_tr, y_tr, binary=False),
        batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
    )
    mc_val = DataLoader(
        NIDSDataset(X_val, y_val, binary=False), batch_size=tcfg["batch_size"], shuffle=False
    )
    if tcfg["use_class_weights"]:
        counts = np.bincount(y_tr, minlength=num_classes).astype(np.float64)
        # Inverse-frequency weights, normalised so the mean weight is ~1.
        weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"[train] class weights = {np.round(weights, 3)}")
    else:
        class_weights = None
    mc_criterion = nn.CrossEntropyLoss(weight=class_weights)
    multiclass_model = train_loop(
        multiclass_model, mc_train, mc_val, mc_criterion, device, tcfg, task="multiclass"
    )

    # ======================= Save =======================
    os.makedirs(abspath(CONFIG["paths"]["models_dir"]), exist_ok=True)
    save_model(
        binary_model, abspath(CONFIG["artifacts"]["binary_model"]),
        input_dim, mcfg["binary_hidden"], mcfg["binary_dropout"], 1, "binary",
    )
    save_model(
        multiclass_model, abspath(CONFIG["artifacts"]["multiclass_model"]),
        input_dim, mcfg["multiclass_hidden"], mcfg["multiclass_dropout"], num_classes, "multiclass",
    )
    print("\n[train] done. Run `python -m src.evaluate` for full test-set metrics.")


if __name__ == "__main__":
    main()
