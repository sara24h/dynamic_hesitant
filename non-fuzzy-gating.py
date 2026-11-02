import os
import json
import time
import random
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from old_dataset_utils import (
    UADFVDataset,
    CustomGenAIDataset,
    NewGenAIDataset,
    create_dataloaders,
    get_sample_info,
)
from visualization_utils import generate_visualizations
from efficiency_utils import (
    Stopwatch,
    summarize_training_times,
    benchmark_efficiency,
    print_headline,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Reproducibility and device
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device():
    """Use one GPU when available; otherwise use CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, using CPU")
    return device


# -----------------------------------------------------------------------------
# Input normalization (kept from the original implementation)
# -----------------------------------------------------------------------------
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float, ...]], stds: List[Tuple[float, ...]]):
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
        mean = getattr(self, f"mean_{idx}")
        std = getattr(self, f"std_{idx}")
        return (x - mean) / std


# -----------------------------------------------------------------------------
# Non-fuzzy gating network
# -----------------------------------------------------------------------------
class NonFuzzyGating(nn.Module):
    """Generate exactly one gating score/weight for each base classifier."""

    def __init__(self, num_models: int, dropout: float = 0.3):
        super().__init__()
        if num_models < 1:
            raise ValueError("num_models must be at least 1")

        # Same feature-extraction architecture as HesitantFuzzyMembership.
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Unlike the fuzzy network, the final layer has num_models outputs,
        # not num_models * num_memberships outputs.
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_models),
        )

    def forward(self, x: torch.Tensor):
        features = self.feature_net(x).flatten(1)
        gate_logits = self.gate(features)             # [B, num_models]
        gate_weights = F.softmax(gate_logits, dim=1)  # [B, num_models]
        return gate_weights, gate_logits


# -----------------------------------------------------------------------------
# Non-fuzzy ensemble with cumulative-weight selection
# -----------------------------------------------------------------------------
class NonFuzzyGatingEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        means: List[Tuple[float, ...]],
        stds: List[Tuple[float, ...]],
        freeze_models: bool = True,
        cum_weight_threshold: float = 0.9,
    ):
        super().__init__()
        if len(models) == 0:
            raise ValueError("At least one base model is required")
        if not (0.0 < cum_weight_threshold <= 1.0):
            raise ValueError("cum_weight_threshold must be in the interval (0, 1]")
        if len(models) != len(means) or len(models) != len(stds):
            raise ValueError("Number of models, means, and stds must match")

        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.gating_network = NonFuzzyGating(num_models=self.num_models)
        self.cum_weight_threshold = float(cum_weight_threshold)

        if freeze_models:
            for model in self.models:
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False

    def train(self, mode: bool = True):
        """Train the gate while keeping frozen base classifiers in eval mode."""
        super().train(mode)
        for model in self.models:
            model.eval()
        return self

    def _compute_mask_vectorized(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Keep the highest-weight models until cumulative mass reaches threshold.

        The model that crosses the threshold is retained. For example, if sorted
        weights are [0.60, 0.30, 0.10] and threshold=0.90, the first two models
        are selected (their cumulative weight is 0.90).
        """
        if gate_weights.ndim != 2 or gate_weights.size(1) != self.num_models:
            raise ValueError("gate_weights must have shape [batch_size, num_models]")

        sorted_weights, sorted_indices = torch.sort(
            gate_weights, dim=1, descending=True
        )
        cumulative_before = torch.cumsum(sorted_weights, dim=1) - sorted_weights

        # Select a classifier when the weight accumulated BEFORE it is below
        # the threshold. This includes the classifier that reaches/crosses it.
        sorted_mask = (cumulative_before < self.cum_weight_threshold).to(
            gate_weights.dtype
        )
        sorted_mask[:, 0] = 1.0  # always retain at least the top classifier

        mask = torch.zeros_like(gate_weights)
        mask.scatter_(1, sorted_indices, sorted_mask)
        return mask

    @staticmethod
    def _ensure_binary_logit_shape(output: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert a binary classifier output to [B, 1], with a clear error otherwise."""
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("A base classifier must return a torch.Tensor")
        if output.ndim == 1:
            output = output.unsqueeze(1)
        else:
            output = output.reshape(output.size(0), -1)
        if output.size(0) != batch_size or output.size(1) != 1:
            raise ValueError(
                "This ensemble expects each base classifier to return one binary logit "
                f"per image, but received shape {tuple(output.shape)}"
            )
        return output

    def forward(self, x: torch.Tensor, return_details: bool = False):
        gate_weights, gate_logits = self.gating_network(x)

        # Non-fuzzy selection: no membership variance and no hesitancy threshold.
        mask = self._compute_mask_vectorized(gate_weights)
        final_weights = gate_weights * mask
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        batch_size = x.size(0)
        outputs = x.new_zeros((batch_size, self.num_models, 1))

        # Avoid running a base classifier if it was not selected for any sample
        # in the current batch.
        active_models = torch.any(final_weights > 0, dim=0).nonzero(as_tuple=True)[0]
        for model_idx_tensor in active_models:
            model_idx = int(model_idx_tensor.item())
            normalized_x = self.normalizations(x, model_idx)
            with torch.no_grad():
                model_output = self.models[model_idx](normalized_x)
            model_output = self._ensure_binary_logit_shape(model_output, batch_size)
            outputs[:, model_idx, :] = model_output

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)

        if return_details:
            # Four outputs intentionally preserve the interface used by the
            # original final-evaluation function:
            # output, final_weights, raw_gate_weights, individual_outputs.
            return final_output, final_weights, gate_weights, outputs
        return final_output, final_weights


# -----------------------------------------------------------------------------
# Base-model loading
# -----------------------------------------------------------------------------
def load_pruned_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError as exc:
        raise ImportError(
            "Cannot import ResNet_50_pruned_hardfakevsreal from model.ResNet_pruned"
        ) from exc

    models = []
    print(f"Loading {len(model_paths)} pruned models...")
    for index, path in enumerate(model_paths, start=1):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Base-model checkpoint does not exist: {path}")

        print(f" [{index}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if "masks" not in checkpoint or "model_state_dict" not in checkpoint:
                raise KeyError(
                    "Checkpoint must contain 'masks' and 'model_state_dict' keys"
                )
            model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint["masks"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device).eval()
            models.append(model)
        except Exception as exc:
            raise RuntimeError(f"Failed to load base model checkpoint '{path}': {exc}") from exc

    if len(models) != len(model_paths):
        raise RuntimeError("Not all requested base models were loaded")
    return models


# -----------------------------------------------------------------------------
# Evaluation of individual base models
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_single_model_exact(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"Evaluating {name}"):
        images = images.to(device)
        labels = labels.to(device).long().reshape(-1)
        images = normalizer(images, 0)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.reshape(images.size(0), -1)
        if output.size(1) != 1:
            raise ValueError(
                f"{name} must return one binary logit per image; got {tuple(output.shape)}"
            )

        predictions = (output[:, 0] > 0).long()
        correct += predictions.eq(labels).sum().item()
        total += labels.numel()

    if total == 0:
        raise RuntimeError(f"No samples were evaluated for {name}")
    accuracy = 100.0 * correct / total
    print(f" {name}: {accuracy:.2f}%")
    return accuracy


@torch.no_grad()
def evaluate_accuracy_and_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device).float().reshape(-1)

        outputs, _ = model(images)
        logits = outputs.reshape(-1)
        loss = criterion(logits, labels)

        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        predictions = (logits > 0).long()
        correct += predictions.eq(labels.long()).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("Validation loader returned no samples")
    return 100.0 * correct / total, total_loss / total


# -----------------------------------------------------------------------------
# Unified final evaluation (adapted from the original script)
# -----------------------------------------------------------------------------
def final_evaluation_unified(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    model_names: List[str],
    args,
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    test_dataset = test_loader.dataset
    num_samples = len(test_dataset)

    lines = [
        "=" * 100,
        "SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):",
        "=" * 100,
        f"{'ID':<10} {'Filename':<60} {'True':<10} {'Pred':<10} {'Correct':<10}",
        "-" * 100,
    ]

    tp = tn = fp = fn = 0
    all_y_true = []
    all_y_score = []
    all_y_pred = []
    all_model_weights = []

    print(f"\nRunning Unified Final Evaluation on {num_samples} samples...")

    for sample_index in tqdm(range(num_samples), desc="Final Eval"):
        try:
            image, label = test_dataset[sample_index]
            original_index = test_dataset.indices[sample_index]
            path, _ = get_sample_info(test_dataset.dataset, original_index)
        except Exception as exc:
            # Do not silently replace unreadable samples with artificial images.
            raise RuntimeError(f"Could not load test sample {sample_index}: {exc}") from exc

        image = image.unsqueeze(0).to(device)
        label_int = int(label)

        with torch.no_grad():
            output, final_weights, _, _ = model(image, return_details=True)
            probability = torch.sigmoid(output.reshape(-1)[0]).item()
        prediction = int(probability > 0.5)

        all_model_weights.append(final_weights.squeeze(0).detach().cpu().numpy())
        all_y_true.append(label_int)
        all_y_score.append(probability)
        all_y_pred.append(prediction)

        if label_int == 1:  # Real = positive class
            if prediction == 1:
                tp += 1
            else:
                fn += 1
        else:  # Fake = negative class
            if prediction == 1:
                fp += 1
            else:
                tn += 1

        filename = os.path.basename(path)
        if len(filename) > 55:
            filename = filename[:25] + "..." + filename[-27:]
        correct_string = "Yes" if prediction == label_int else "No"
        lines.append(
            f"{sample_index + 1:<10} {filename:<60} {label_int:<10} "
            f"{prediction:<10} {correct_string:<10}"
        )

    total_samples = len(all_y_true)
    if total_samples == 0:
        raise RuntimeError("No test samples were successfully evaluated")

    correct_count = tp + tn
    accuracy = 100.0 * correct_count / total_samples

    precision_real = tp / (tp + fp) if (tp + fp) else 0.0
    recall_real = tp / (tp + fn) if (tp + fn) else 0.0
    f1_real = (
        2 * precision_real * recall_real / (precision_real + recall_real)
        if (precision_real + recall_real)
        else 0.0
    )

    precision_fake = tn / (tn + fn) if (tn + fn) else 0.0
    recall_fake = tn / (tn + fp) if (tn + fp) else 0.0
    f1_fake = (
        2 * precision_fake * recall_fake / (precision_fake + recall_fake)
        if (precision_fake + recall_fake)
        else 0.0
    )

    macro_precision = (precision_real + precision_fake) / 2.0
    macro_recall = (recall_real + recall_fake) / 2.0
    macro_f1 = (f1_real + f1_fake) / 2.0
    balanced_accuracy = 100.0 * (recall_real + recall_fake) / 2.0

    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Fake", "Predicted Real"],
        yticklabels=["Actual Fake", "Actual Real"],
        cbar=True,
        annot_kws={"size": 16},
    )
    plt.title(
        f"Confusion Matrix\nAccuracy: {accuracy:.2f}% | "
        f"Balanced Acc: {balanced_accuracy:.2f}% | Macro-F1: {macro_f1:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    confusion_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {confusion_path}")

    print("\n" + "=" * 70)
    print("FINAL NON-FUZZY GATING RESULTS")
    print("=" * 70)
    print(f"Accuracy          : {accuracy:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 45)
    print(f"{'Fake (0)':<12} {precision_fake:10.4f} {recall_fake:10.4f} {f1_fake:10.4f}")
    print(f"{'Real (1)':<12} {precision_real:10.4f} {recall_real:10.4f} {f1_real:10.4f}")
    print("-" * 45)
    print(f"{'Macro Avg':<12} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f}")
    print(f"Correct Predictions  : {correct_count} ({accuracy:.2f}%)")
    print(f"Incorrect Predictions: {total_samples - correct_count} "
          f"({100.0 - accuracy:.2f}%)")
    print("=" * 70)

    # Per-classifier activation statistics
    weights_np = np.asarray(all_model_weights)
    activation_threshold = 1e-4
    active_mask = (weights_np > activation_threshold).astype(np.float32)
    activation_rates = active_mask.mean(axis=0) * 100.0
    average_weights = weights_np.mean(axis=0)
    maximum_weights = weights_np.max(axis=0)
    active_per_sample = active_mask.sum(axis=1)

    print("\nMODEL ACTIVATION STATISTICS")
    print("-" * 70)
    for index, model_name in enumerate(model_names):
        print(
            f"{index + 1:2d}. {model_name:<30}: "
            f"{activation_rates[index]:6.2f}% | "
            f"Avg={average_weights[index]:.6f} | Max={maximum_weights[index]:.6f}"
        )
    print(f"Mean active models per sample: {active_per_sample.mean():.3f}")
    print(f"Min/Max active models: {active_per_sample.min()} / {active_per_sample.max()}")

    unique_counts, count_frequencies = np.unique(
        active_per_sample.astype(int), return_counts=True
    )
    activation_stats = {
        "activation_threshold": activation_threshold,
        "activation_rates_pct": {
            name: float(activation_rates[i]) for i, name in enumerate(model_names)
        },
        "average_weights": {
            name: float(average_weights[i]) for i, name in enumerate(model_names)
        },
        "max_weights": {
            name: float(maximum_weights[i]) for i, name in enumerate(model_names)
        },
        "active_per_sample_stats": {
            "min": int(active_per_sample.min()),
            "max": int(active_per_sample.max()),
            "mean": float(active_per_sample.mean()),
            "median": float(np.median(active_per_sample)),
        },
        "active_model_count_distribution": {
            str(int(count)): int(freq)
            for count, freq in zip(unique_counts, count_frequencies)
        },
        "per_sample_weights": weights_np.tolist(),
    }
    with open(
        os.path.join(save_dir, "model_activation_stats.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(activation_stats, file, indent=2, ensure_ascii=False)

    summary_lines = [
        "-" * 100,
        "SUMMARY STATISTICS:",
        "-" * 100,
        f"Overall Accuracy: {accuracy:.2f}%",
        f"Balanced Accuracy: {balanced_accuracy:.2f}%",
        f"Macro F1-Score: {macro_f1:.4f}",
        "\n--- Real (1) Class Metrics ---",
        f"Precision (Real): {precision_real:.4f}",
        f"Recall (Real): {recall_real:.4f}",
        f"F1-Score (Real): {f1_real:.4f}",
        "\n--- Fake (0) Class Metrics ---",
        f"Precision (Fake): {precision_fake:.4f}",
        f"Recall (Fake): {recall_fake:.4f}",
        f"F1-Score (Fake): {f1_fake:.4f}",
        f"\nCorrect Predictions: {correct_count} ({accuracy:.2f}%)",
        f"Incorrect Predictions: {total_samples - correct_count} "
        f"({100.0 - accuracy:.2f}%)",
        "\n".join(lines),
    ]
    with open(os.path.join(save_dir, "prediction_log.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))

    # ROC data (same format as the original pipeline)
    y_true_np = np.asarray(all_y_true, dtype=np.int64)
    y_score_np = np.asarray(all_y_score, dtype=np.float32)
    y_pred_np = np.asarray(all_y_pred, dtype=np.int64)
    roc_data = {
        "metadata": {
            "seed": args.seed,
            "dataset": args.dataset,
            "num_samples": int(total_samples),
            "positive_count": int(np.sum(y_true_np)),
            "negative_count": int(total_samples - np.sum(y_true_np)),
            "model": "non_fuzzy_gating_ensemble",
        },
        "y_true": y_true_np.tolist(),
        "y_score": y_score_np.tolist(),
        "y_pred": y_pred_np.tolist(),
    }
    with open(os.path.join(save_dir, "roc_data_test.json"), "w", encoding="utf-8") as file:
        json.dump(roc_data, file, indent=2, ensure_ascii=False)
    with open(os.path.join(save_dir, "roc_data_test.txt"), "w", encoding="utf-8") as file:
        file.write("y_true\ty_score\ty_pred\n")
        for true_label, score, prediction in zip(y_true_np, y_score_np, y_pred_np):
            file.write(f"{int(true_label)}\t{float(score):.6f}\t{int(prediction)}\n")

    # Keep the visualization call from the original script.
    visualization_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)
    try:
        generate_visualizations(
            model,
            test_loader,
            device,
            visualization_dir,
            model_names,
            args.num_grad_cam_samples,
            args.num_lime_samples,
            args.dataset,
            is_main=True,
        )
    except Exception as exc:
        # Keep metric evaluation/results even if an optional visualization
        # utility assumes attributes specific to the original fuzzy model.
        print(f"WARNING: visualization generation failed: {exc}")

    return accuracy, y_true_np, y_score_np


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_non_fuzzy_gating(
    ensemble_model: NonFuzzyGatingEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str,
    model_names: List[str],
):
    os.makedirs(save_dir, exist_ok=True)

    gate_network = ensemble_model.gating_network
    optimizer = torch.optim.AdamW(
        gate_network.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_accuracy = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "mean_active_models": [],
        "mean_gate_entropy": [],
    }

    epoch_times, train_times, val_times = [], [], []
    total_stopwatch = Stopwatch(device).start()

    print("=" * 70)
    print("TRAINING NON-FUZZY GATING NETWORK")
    print("=" * 70)
    trainable_parameters = sum(p.numel() for p in gate_network.parameters() if p.requires_grad)
    print(f"Trainable gate parameters: {trainable_parameters:,}")
    print(f"Epochs: {num_epochs} | Initial LR: {learning_rate}")
    print(f"Number of base classifiers: {len(model_names)}")
    print(f"Cumulative selection threshold: {ensemble_model.cum_weight_threshold}")

    for epoch in range(num_epochs):
        epoch_stopwatch = Stopwatch(device).start()
        train_stopwatch = Stopwatch(device).start()
        ensemble_model.train()

        train_loss_sum = 0.0
        train_correct = 0
        samples_seen = 0
        active_models_sum = 0.0
        gate_entropy_sum = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device).float().reshape(-1)
            optimizer.zero_grad(set_to_none=True)

            outputs, final_weights, raw_gate_weights, _ = ensemble_model(
                images, return_details=True
            )
            logits = outputs.reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.numel()
            train_loss_sum += loss.item() * batch_size
            predictions = (logits.detach() > 0).long()
            train_correct += predictions.eq(labels.long()).sum().item()
            samples_seen += batch_size

            active_count = (final_weights.detach() > 1e-4).sum(dim=1).float()
            active_models_sum += active_count.sum().item()
            entropy = -(
                raw_gate_weights.detach()
                * raw_gate_weights.detach().clamp_min(1e-8).log()
            ).sum(dim=1)
            gate_entropy_sum += entropy.sum().item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if samples_seen == 0:
            raise RuntimeError("Training loader returned no samples")

        train_time = train_stopwatch.stop()
        train_loss = train_loss_sum / samples_seen
        train_accuracy = 100.0 * train_correct / samples_seen
        mean_active_models = active_models_sum / samples_seen
        mean_gate_entropy = gate_entropy_sum / samples_seen

        validation_stopwatch = Stopwatch(device).start()
        val_accuracy, val_loss = evaluate_accuracy_and_loss(
            ensemble_model, val_loader, device, criterion
        )
        validation_time = validation_stopwatch.stop()
        epoch_time = epoch_stopwatch.stop()

        epoch_times.append(epoch_time)
        train_times.append(train_time)
        val_times.append(validation_time)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)
        history["mean_active_models"].append(mean_active_models)
        history["mean_gate_entropy"].append(mean_gate_entropy)

        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 70)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"Mean active models/sample: {mean_active_models:.3f}")
        print(f"Mean raw-gate entropy: {mean_gate_entropy:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.7f}")
        print(
            f"Time: epoch {epoch_time:.1f}s "
            f"(train {train_time:.1f}s | validation {validation_time:.1f}s)"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(save_dir, "best_non_fuzzy_gating.pt")
            torch.save(
                {"gating_state_dict": gate_network.state_dict()}, checkpoint_path
            )
            print(f"Best gate checkpoint saved: {checkpoint_path}")

    total_training_time = total_stopwatch.stop()
    timing = summarize_training_times(
        epoch_times, train_times, val_times, total_training_time
    )
    print(
        f"Total training time: {total_training_time / 60.0:.2f} min "
        f"(mean epoch: {timing['mean_epoch_time_s']:.1f}s)"
    )

    # Training curves
    epochs_axis = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, history["train_loss"], label="train", linewidth=2)
    plt.plot(epochs_axis, history["val_loss"], label="validation", linewidth=2)
    plt.title("Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, history["train_acc"], label="train", linewidth=2)
    plt.plot(epochs_axis, history["val_acc"], label="validation", linewidth=2)
    plt.title("Accuracy", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    curves_path = os.path.join(save_dir, "training_curves_non_fuzzy.png")
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {curves_path}")

    with open(os.path.join(save_dir, "training_history_non_fuzzy.json"), "w") as file:
        json.dump(history, file, indent=2)

    return best_val_accuracy, timing


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Non-Fuzzy Gating Ensemble Training (Single GPU)"
    )

    # Same principal arguments as the original script
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "wild",
            "artiface",
            "hard_fake_real",
            "deepfake_lab",
            "uadfV",
            "custom_genai",
            "custom_genai_v2",
            "real_fake_dataset",
            "20k",
        ],
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_paths", type=str, nargs="+", required=True)
    parser.add_argument("--model_names", type=str, nargs="+", required=True)
    parser.add_argument("--save_dir", type=str, default="./output_non_fuzzy")
    parser.add_argument("--seed", type=int, default=42)

    # Non-fuzzy ensemble selection argument
    parser.add_argument("--cum_weight_threshold", type=float, default=0.9)

    # Visualization arguments retained from the original script
    parser.add_argument("--num_grad_cam_samples", type=int, default=5)
    parser.add_argument("--num_lime_samples", type=int, default=5)

    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")
    if not (0.0 < args.cum_weight_threshold <= 1.0):
        raise ValueError("--cum_weight_threshold must be in (0, 1]")

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = setup_device()
    pipeline_stopwatch = Stopwatch(device).start()

    print("=" * 70)
    print("NON-FUZZY GATING ENSEMBLE")
    print(f"Single GPU Mode | Seed: {args.seed}")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of base models: {len(args.model_paths)}")
    print(f"Cumulative weight threshold: {args.cum_weight_threshold}")
    print("=" * 70 + "\n")

    # Same normalization values as the original script
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
    model_names = list(args.model_names)

    ensemble = NonFuzzyGatingEnsemble(
        models=base_models,
        means=means,
        stds=stds,
        freeze_models=True,
        cum_weight_threshold=args.cum_weight_threshold,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel() for parameter in ensemble.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in ensemble.parameters())
    print(
        f"Total params: {total_parameters:,} | Trainable: {trainable_parameters:,} | "
        f"Frozen: {total_parameters - trainable_parameters:,}\n"
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        args.batch_size,
        dataset_type=args.dataset,
        is_distributed=False,
        seed=args.seed,
        is_main=True,
    )

    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL PERFORMANCE (Before Training)")
    print("=" * 70)

    individual_accuracies = []
    for index, model in enumerate(base_models):
        accuracy = evaluate_single_model_exact(
            model,
            test_loader,
            device,
            f"Model {index + 1} ({model_names[index]})",
            means[index],
            stds[index],
        )
        individual_accuracies.append(accuracy)

    best_single_accuracy = max(individual_accuracies)
    best_single_index = int(np.argmax(individual_accuracies))
    print(
        f"\nBest Single: Model {best_single_index + 1} "
        f"({model_names[best_single_index]}) -> {best_single_accuracy:.2f}%"
    )
    print("=" * 70)

    best_val_accuracy, train_timing = train_non_fuzzy_gating(
        ensemble,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
        args.save_dir,
        model_names,
    )

    # Restore the best gate based on validation accuracy
    best_checkpoint_path = os.path.join(args.save_dir, "best_non_fuzzy_gating.pt")
    if not os.path.isfile(best_checkpoint_path):
        raise FileNotFoundError(f"Best gate checkpoint not found: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    ensemble.gating_network.load_state_dict(checkpoint["gating_state_dict"])
    print("Best non-fuzzy gate checkpoint loaded.\n")

    print("\n" + "=" * 70)
    print("FINAL NON-FUZZY ENSEMBLE EVALUATION")
    print("=" * 70)

    efficiency = benchmark_efficiency(
        ensemble,
        test_loader,
        device,
        model_names,
        latency_samples=len(test_loader.dataset),
    )
    efficiency["training_time"] = train_timing
    print_headline(efficiency)

    evaluation_stopwatch = Stopwatch(device).start()
    final_accuracy, y_true, y_scores = final_evaluation_unified(
        ensemble, test_loader, device, args.save_dir, model_names, args
    )
    final_evaluation_time = evaluation_stopwatch.stop()

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Best Single Model: {best_single_accuracy:.2f}%")
    print(f"Non-Fuzzy Gating Ensemble: {final_accuracy:.2f}%")
    print(f"Improvement: {final_accuracy - best_single_accuracy:+.2f}%")
    print("=" * 70)

    final_results = {
        "seed": args.seed,
        "method": "Non_Fuzzy_Gating",
        "best_single_model": {
            "name": model_names[best_single_index],
            "accuracy": float(best_single_accuracy),
        },
        "ensemble": {"test_accuracy": float(final_accuracy)},
        "improvement": float(final_accuracy - best_single_accuracy),
        "best_validation_accuracy": float(best_val_accuracy),
        "cum_weight_threshold": float(args.cum_weight_threshold),
    }
    with open(os.path.join(args.save_dir, "final_results_non_fuzzy.json"), "w") as file:
        json.dump(final_results, file, indent=4)

    torch.save(
        {
            "ensemble_state_dict": ensemble.state_dict(),
            "gating_state_dict": ensemble.gating_network.state_dict(),
            "model_names": model_names,
            "means": means,
            "stds": stds,
            "seed": args.seed,
            "cum_weight_threshold": args.cum_weight_threshold,
            "method": "Non_Fuzzy_Gating",
        },
        os.path.join(args.save_dir, "final_non_fuzzy_gating_ensemble.pt"),
    )

    efficiency["final_evaluation_time_s"] = final_evaluation_time
    efficiency["total_pipeline_time_s"] = pipeline_stopwatch.stop()
    with open(os.path.join(args.save_dir, "efficiency_report_non_fuzzy.json"), "w") as file:
        json.dump(efficiency, file, indent=2)
    print(
        "Efficiency report saved: "
        f"{os.path.join(args.save_dir, 'efficiency_report_non_fuzzy.json')}"
    )


if __name__ == "__main__":
    main()
