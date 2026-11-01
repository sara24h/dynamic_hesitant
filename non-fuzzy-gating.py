import argparse
import csv
import json
import os
import random
import warnings
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from old_dataset_utils import create_dataloaders, get_sample_info

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA is not available; using CPU")
    return device


# ---------------------------------------------------------------------------
# Per-classifier input normalization (same as the supplied fuzzy code)
# ---------------------------------------------------------------------------
class MultiModelNormalization(nn.Module):
    def __init__(self, means: Sequence[Tuple[float, float, float]],
                 stds: Sequence[Tuple[float, float, float]]):
        super().__init__()
        if len(means) != len(stds):
            raise ValueError("means and stds must have the same length")
        for i, (mean, std) in enumerate(zip(means, stds)):
            self.register_buffer(
                f"mean_{i}", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.register_buffer(
                f"std_{i}", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
            )

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f"mean_{idx}")) / getattr(self, f"std_{idx}")


# ---------------------------------------------------------------------------
# Non-fuzzy multi-head gating network
# ---------------------------------------------------------------------------
class NonFuzzyMultiHeadGating(nn.Module):
    """Conventional multi-head gating counterpart.

    The feature extractor and trainable layer dimensions are identical to
    HesitantFuzzyMembership in the supplied code. The final layer returns
    K ordinary gating-logit vectors, each containing one logit per expert.
    A softmax is applied over experts independently for each head. The K
    distributions are then combined using learned head weights.

    Unlike the fuzzy branch, these values are not independent fuzzy
    memberships: each head is a standard categorical gate whose probabilities
    sum to one over classifiers.
    """

    def __init__(self, num_models: int, num_heads: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        if num_models < 1 or num_heads < 1:
            raise ValueError("num_models and num_heads must both be >= 1")
        self.num_models = int(num_models)
        self.num_heads = int(num_heads)

        # Exact same feature extractor as HesitantFuzzyMembership.
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Same dimensions as the fuzzy membership_generator:
        # Linear(128,128) -> Linear(128,num_models*num_heads).
        self.gating_generator = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_models * self.num_heads),
        )

        # Same number of trainable aggregation parameters as the fuzzy model.
        self.head_aggregation_logits = nn.Parameter(
            torch.ones(self.num_heads, dtype=torch.float32) / self.num_heads
        )

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        logits = self.gating_generator(features)
        # [B, M*K] -> [B, K, M]
        head_logits = logits.view(-1, self.num_models, self.num_heads).transpose(1, 2)

        # Conventional categorical gating: every head distributes unit mass
        # across the M classifiers.
        head_weights = F.softmax(head_logits, dim=2)  # [B, K, M]

        # Learned aggregation across the K ordinary gating heads.
        head_alpha = F.softmax(self.head_aggregation_logits, dim=0)  # [K]
        final_weights = (
            head_weights * head_alpha.view(1, self.num_heads, 1)
        ).sum(dim=1)  # [B, M], sums to 1

        # Analogue of the fuzzy variance-based hesitancy statistic:
        # variability of each expert's gate probability across K heads,
        # averaged over experts. Shape: [B].
        disagreement = head_weights.var(dim=1, unbiased=False).mean(dim=1)
        return final_weights, head_weights, disagreement


# ---------------------------------------------------------------------------
# Ensemble: same frozen experts, normalization, thresholding, and fusion
# ---------------------------------------------------------------------------
class NonFuzzyGatingEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        means: Sequence[Tuple[float, float, float]],
        stds: Sequence[Tuple[float, float, float]],
        num_heads: int = 3,
        dropout: float = 0.3,
        freeze_models: bool = True,
        cum_weight_threshold: float = 0.9,
        hesitancy_threshold: float = 0.2,
    ):
        super().__init__()
        if len(models) == 0:
            raise ValueError("At least one base classifier must be supplied")
        if len(means) != len(models) or len(stds) != len(models):
            raise ValueError("One mean/std tuple is required for each base classifier")
        if not (0.0 < cum_weight_threshold <= 1.0):
            raise ValueError("cum_weight_threshold must be in (0, 1]")

        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating = NonFuzzyMultiHeadGating(
            num_models=self.num_models,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cum_weight_threshold = float(cum_weight_threshold)
        self.hesitancy_threshold = float(hesitancy_threshold)

        if freeze_models:
            for model in self.models:
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep base classifiers frozen and in evaluation mode, as in the fuzzy code.
        for model in self.models:
            model.eval()
        return self

    def _compute_mask(self, weights: torch.Tensor,
                      disagreement: torch.Tensor) -> torch.Tensor:
        sorted_weights, sorted_indices = torch.sort(
            weights, dim=1, descending=True
        )
        cumulative_weights = torch.cumsum(sorted_weights, dim=1)
        mask_sorted = (cumulative_weights <= self.cum_weight_threshold).float()
        mask_sorted[:, 0] = 1.0  # Always retain at least the top-weight classifier.

        # Matched fallback rule: retain all classifiers when gate-head
        # disagreement exceeds the same numerical threshold used by HF.
        high_disagreement = (disagreement > self.hesitancy_threshold).unsqueeze(1)
        mask_sorted = torch.where(
            high_disagreement, torch.ones_like(mask_sorted), mask_sorted
        )

        mask = torch.zeros_like(weights)
        mask.scatter_(1, sorted_indices, mask_sorted)
        return mask

    def forward(self, x: torch.Tensor, return_details: bool = False):
        weights, head_weights, disagreement = self.gating(x)
        mask = self._compute_mask(weights, disagreement)
        selected_weights = weights * mask
        selected_weights = selected_weights / (
            selected_weights.sum(dim=1, keepdim=True) + 1e-8
        )

        batch_size = x.size(0)
        outputs = x.new_zeros((batch_size, self.num_models, 1))
        active_models = torch.any(selected_weights > 0.0, dim=0).nonzero(
            as_tuple=True
        )[0]

        for model_idx in active_models:
            idx = int(model_idx.item())
            normalized_x = self.normalizations(x, idx)
            with torch.no_grad():
                model_output = self.models[idx](normalized_x)
                if isinstance(model_output, (tuple, list)):
                    model_output = model_output[0]
                model_output = model_output.reshape(batch_size, -1)[:, :1]
            outputs[:, idx, :] = model_output

        final_output = (outputs * selected_weights.unsqueeze(-1)).sum(dim=1)
        if return_details:
            return final_output, selected_weights, disagreement, outputs, head_weights
        return final_output, selected_weights


# ---------------------------------------------------------------------------
# Load the same pruned base classifiers as the fuzzy experiment
# ---------------------------------------------------------------------------
def load_pruned_models(model_paths: Sequence[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError as exc:
        raise ImportError(
            "Could not import model.ResNet_pruned.ResNet_50_pruned_hardfakevsreal. "
            "Run this script from the same project root as the original fuzzy code."
        ) from exc

    models: List[nn.Module] = []
    for i, path in enumerate(model_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Base-model checkpoint not found: {path}")
        print(f"Loading base model {i + 1}/{len(model_paths)}: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Compatibility with older PyTorch versions.
            checkpoint = torch.load(path, map_location="cpu")

        if "masks" not in checkpoint or "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint {path} must contain 'masks' and 'model_state_dict'."
            )
        model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint["masks"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_accuracy_and_loss(model: nn.Module, loader, device: torch.device,
                                criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)
        outputs, _ = model(images)
        logits = outputs.view(-1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.numel()
        predictions = (logits > 0).long()
        correct += predictions.eq(labels.long()).sum().item()
        total += labels.numel()
    if total == 0:
        raise RuntimeError("Validation loader produced zero samples")
    return 100.0 * correct / total, total_loss / total


def train_non_fuzzy(
    ensemble: NonFuzzyGatingEnsemble,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str,
    args,
):
    os.makedirs(save_dir, exist_ok=True)
    gating_net = ensemble.gating
    optimizer = torch.optim.AdamW(
        gating_net.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_epoch = -1
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    print("\n" + "=" * 76)
    print("TRAINING NON-FUZZY MULTI-HEAD GATING BASELINE")
    print("=" * 76)
    print(f"Trainable gating parameters: {sum(p.numel() for p in gating_net.parameters() if p.requires_grad):,}")
    print(f"Epochs={num_epochs} | LR={learning_rate} | Batch size={args.batch_size}")
    print(f"Cumulative threshold={ensemble.cum_weight_threshold}")
    print(f"Disagreement threshold={ensemble.hesitancy_threshold}")
    print(f"Number of conventional gate heads={gating_net.num_heads}")

    for epoch in range(num_epochs):
        ensemble.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)

            optimizer.zero_grad(set_to_none=True)
            outputs, _ = ensemble(images)
            logits = outputs.view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.numel()
            running_loss += loss.item() * batch_count
            running_correct += (logits.detach() > 0).long().eq(labels.long()).sum().item()
            seen += batch_count
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if seen == 0:
            raise RuntimeError("Training loader produced zero samples")
        train_loss = running_loss / seen
        train_acc = 100.0 * running_correct / seen
        val_acc, val_loss = evaluate_accuracy_and_loss(
            ensemble, val_loader, device, criterion
        )
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train loss {train_loss:.5f} | Train acc {train_acc:.2f}% | "
            f"Val loss {val_loss:.5f} | Val acc {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "gating_state_dict": gating_net.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": float(best_val_acc),
                    "args": vars(args),
                },
                os.path.join(save_dir, "best_non_fuzzy_gating.pt"),
            )

    # Save training history and curves.
    with open(os.path.join(save_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs = np.arange(1, num_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close(fig)

    return best_val_acc, best_epoch


# ---------------------------------------------------------------------------
# Test evaluation: sample order and per-sample predictions are saved for
# paired comparisons (e.g., McNemar) against the fuzzy model.
# ---------------------------------------------------------------------------
def _get_filename(test_dataset, sample_index: int) -> str:
    try:
        original_index = test_dataset.indices[sample_index]
        base_dataset = test_dataset.dataset
        path, _ = get_sample_info(base_dataset, original_index)
        return os.path.basename(str(path))
    except Exception:
        return f"sample_{sample_index:06d}"


@torch.no_grad()
def evaluate_test(
    model: NonFuzzyGatingEnsemble,
    test_loader,
    device: torch.device,
    save_dir: str,
    model_names: Sequence[str],
    args,
):
    model.eval()
    test_dataset = test_loader.dataset
    y_true, y_score, y_pred = [], [], []
    filenames, per_sample_weights, per_sample_disagreement = [], [], []

    print(f"\nFinal test evaluation: {len(test_dataset)} samples")
    for sample_index in tqdm(range(len(test_dataset)), desc="Test evaluation"):
        image, label = test_dataset[sample_index]
        image = image.unsqueeze(0).to(device)
        label_int = int(label.item()) if torch.is_tensor(label) else int(label)

        output, weights, disagreement, _, _ = model(image, return_details=True)
        probability = torch.sigmoid(output.reshape(-1)[0]).item()
        prediction = int(probability > 0.5)

        y_true.append(label_int)
        y_score.append(float(probability))
        y_pred.append(prediction)
        filenames.append(_get_filename(test_dataset, sample_index))
        per_sample_weights.append(weights.squeeze(0).cpu().tolist())
        per_sample_disagreement.append(float(disagreement.reshape(-1)[0].cpu().item()))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    weights_np = np.asarray(per_sample_weights, dtype=np.float64)

    if y_true_np.size == 0:
        raise RuntimeError("Test loader produced zero samples")

    # Label convention retained from the supplied project: Real=1, Fake=0.
    tp = int(np.sum((y_true_np == 1) & (y_pred_np == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred_np == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    accuracy = 100.0 * (tp + tn) / len(y_true_np)
    precision_real = tp / (tp + fp) if tp + fp else 0.0
    recall_real = tp / (tp + fn) if tp + fn else 0.0
    f1_real = (
        2 * precision_real * recall_real / (precision_real + recall_real)
        if precision_real + recall_real else 0.0
    )
    precision_fake = tn / (tn + fn) if tn + fn else 0.0
    recall_fake = tn / (tn + fp) if tn + fp else 0.0
    f1_fake = (
        2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        if precision_fake + recall_fake else 0.0
    )
    macro_f1 = (f1_real + f1_fake) / 2.0
    balanced_accuracy = 100.0 * (recall_real + recall_fake) / 2.0

    active_mask = (weights_np > 1e-4).astype(np.int64)
    activation_rates = active_mask.mean(axis=0) * 100.0
    mean_weights = weights_np.mean(axis=0)
    active_per_sample = active_mask.sum(axis=1)

    print("\n" + "=" * 72)
    print("NON-FUZZY MULTI-HEAD GATING — TEST RESULTS")
    print("=" * 72)
    print(f"Accuracy          : {accuracy:.2f}%")
    print(f"Balanced accuracy : {balanced_accuracy:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"TP={tp} | TN={tn} | FP={fp} | FN={fn}")
    print(f"Mean active classifiers/sample: {active_per_sample.mean():.3f}")
    for i, name in enumerate(model_names):
        print(
            f"{name:<32} activation={activation_rates[i]:6.2f}% "
            f"mean_weight={mean_weights[i]:.6f}"
        )
    print("=" * 72)

    # Confusion matrix.
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted Fake", "Predicted Real"])
    ax.set_yticks([0, 1], labels=["Actual Fake", "Actual Real"])
    ax.set_title(f"Non-fuzzy gating | Accuracy={accuracy:.2f}%")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix_non_fuzzy.png"), dpi=300)
    plt.close(fig)

    # Per-sample file supports paired tests against the fuzzy predictions.
    prediction_csv = os.path.join(save_dir, "sample_predictions_non_fuzzy.csv")
    with open(prediction_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "filename", "y_true", "y_score", "y_pred", "correct"])
        for i, (name, truth, score, pred) in enumerate(
            zip(filenames, y_true, y_score, y_pred)
        ):
            writer.writerow([i, name, truth, f"{score:.10f}", pred, int(truth == pred)])

    np.savez_compressed(
        os.path.join(save_dir, "roc_data_test_non_fuzzy.npz"),
        y_true=y_true_np,
        y_score=y_score_np,
        y_pred=y_pred_np,
    )

    activation_stats = {
        "seed": int(args.seed),
        "dataset": args.dataset,
        "activation_rates_pct": {
            name: float(activation_rates[i]) for i, name in enumerate(model_names)
        },
        "average_weights": {
            name: float(mean_weights[i]) for i, name in enumerate(model_names)
        },
        "active_per_sample": {
            "min": int(active_per_sample.min()),
            "max": int(active_per_sample.max()),
            "mean": float(active_per_sample.mean()),
            "median": float(np.median(active_per_sample)),
        },
        "per_sample_weights": weights_np.tolist(),
        "per_sample_disagreement": per_sample_disagreement,
        "filenames_in_test_order": filenames,
    }
    with open(os.path.join(save_dir, "model_activation_stats_non_fuzzy.json"), "w", encoding="utf-8") as f:
        json.dump(activation_stats, f, indent=2, ensure_ascii=False)

    results = {
        "method": "Non_Fuzzy_MultiHead_Gating",
        "seed": int(args.seed),
        "dataset": args.dataset,
        "num_samples": int(len(y_true_np)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "best_epoch": int(args.best_epoch),
        "num_trainable_gating_parameters": int(
            sum(p.numel() for p in model.gating.parameters() if p.requires_grad)
        ),
        "num_heads": int(args.num_heads),
        "cum_weight_threshold": float(args.cum_weight_threshold),
        "hesitancy_threshold": float(args.hesitancy_threshold),
    }
    with open(os.path.join(save_dir, "final_results_non_fuzzy.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reviewer-required non-fuzzy multi-head gating baseline"
    )
    # Keep these identical to the supplied fuzzy script unless the fuzzy run
    # used different values; in that case pass the exact same values here.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["wild", "artiface", "hard_fake_real", "deepfake_lab",
                                 "uadfV", "custom_genai", "custom_genai_v2",
                                 "real_fake_dataset", "20k"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--model_names", type=str, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="./ablation_non_fuzzy")
    parser.add_argument("--seed", type=int, default=42)

    # Must match num_memberships in the fuzzy experiment.
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    # Must match the actual values used by the fuzzy experiment for this dataset.
    parser.add_argument("--cum_weight_threshold", type=float, default=0.9)
    parser.add_argument("--hesitancy_threshold", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.model_names) != len(args.model_paths):
        raise ValueError("The number of --model_names must equal --model_paths")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = setup_device()

    print("\n" + "=" * 76)
    print("NON-FUZZY MULTI-HEAD GATING BASELINE")
    print("=" * 76)
    print(f"Dataset={args.dataset} | Seed={args.seed}")
    print(f"Epochs={args.epochs} | LR={args.lr} | Batch={args.batch_size}")
    print(f"Heads={args.num_heads} | Dropout={args.dropout}")
    print(f"Cumulative threshold={args.cum_weight_threshold}")
    print(f"Hesitancy/disagreement threshold={args.hesitancy_threshold}")

    default_means = [
        (0.5207, 0.4258, 0.3806),
        (0.4460, 0.3622, 0.3416),
        (0.4668, 0.3816, 0.3414),
    ]
    default_stds = [
        (0.2490, 0.2239, 0.2212),
        (0.2057, 0.1849, 0.1761),
        (0.2410, 0.2161, 0.2081),
    ]
    num_models = len(args.model_paths)
    if num_models > len(default_means):
        means = default_means + [default_means[-1]] * (num_models - len(default_means))
        stds = default_stds + [default_stds[-1]] * (num_models - len(default_stds))
    else:
        means = default_means[:num_models]
        stds = default_stds[:num_models]

    base_models = load_pruned_models(args.model_paths, device)
    model_names = args.model_names[:len(base_models)]

    ensemble = NonFuzzyGatingEnsemble(
        models=base_models,
        means=means,
        stds=stds,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
        hesitancy_threshold=args.hesitancy_threshold,
    ).to(device)

    trainable_count = sum(
        parameter.numel() for parameter in ensemble.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in ensemble.parameters())
    frozen_count = total_count - trainable_count
    print(f"Total parameters={total_count:,} | Trainable={trainable_count:,} | Frozen={frozen_count:,}")
    print(
        "Parameter-parity note: with the same M, K, feature extractor, hidden width, "
        "and dropout as HesitantFuzzyMembership, the trainable gating parameter "
        "count is exactly matched (not padded with unused parameters)."
    )

    # Identical dataloader construction and seed are required for the paired run.
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        args.batch_size,
        dataset_type=args.dataset,
        is_distributed=False,
        seed=args.seed,
        is_main=True,
    )

    best_val_acc, best_epoch = train_non_fuzzy(
        ensemble=ensemble,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        args=args,
    )
    args.best_epoch = best_epoch

    checkpoint_path = os.path.join(args.save_dir, "best_non_fuzzy_gating.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint was not created: {checkpoint_path}")
    try:
        best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
    ensemble.gating.load_state_dict(best_checkpoint["gating_state_dict"])

    results = evaluate_test(
        model=ensemble,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        model_names=model_names,
        args=args,
    )
    results["best_validation_accuracy"] = float(best_val_acc)
    results["best_epoch"] = int(best_epoch)
    with open(os.path.join(args.save_dir, "final_results_non_fuzzy.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted. Results are saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
