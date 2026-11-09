"""Evaluation utilities for model-competence and variance/competence analysis.

Designed to be added to the user's HFDE training/evaluation script.

Expected ensemble interface:
    output, final_weights, memberships, outputs = ensemble(image, return_details=True)
    logits = ensemble.compute_all_model_logits(image)

Labels must be binary: 0 = fake, 1 = real. Base-model and ensemble outputs
are logits, so a prediction is positive/real when logit > 0.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _to_python_float(value: Any) -> Optional[float]:
    """Convert a scalar to a finite JSON-safe float; return None for NaN/Inf."""
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_correlation(
    x: Sequence[float], y: Sequence[float], method: str = "spearman"
) -> Tuple[Optional[float], Optional[float]]:
    """Correlation that safely handles constant/too-short arrays."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None, None

    try:
        if method == "pearson":
            result = pearsonr(x, y)
        elif method == "spearman":
            result = spearmanr(x, y)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
        return _to_python_float(result.statistic), _to_python_float(result.pvalue)
    except (ValueError, FloatingPointError):
        return None, None


def _safe_auc(binary_target: Sequence[int], score: Sequence[float]) -> Optional[float]:
    """AUROC, or None when the target contains only one class."""
    y = np.asarray(binary_target, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    valid = np.isfinite(y) & np.isfinite(s)
    y, s = y[valid], s[valid]
    if len(y) == 0 or len(np.unique(y)) != 2:
        return None
    try:
        return _to_python_float(roc_auc_score(y, s))
    except ValueError:
        return None


def _bootstrap_indices(
    n: int,
    rng: np.random.Generator,
    group_ids: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Sample rows, or sample groups (cluster bootstrap) with replacement."""
    if group_ids is None:
        return rng.integers(0, n, size=n)

    groups = np.asarray([str(x) for x in group_ids], dtype=object)
    if len(groups) != n:
        raise ValueError(
            f"group_ids has {len(groups)} entries, but there are {n} samples"
        )

    unique_groups = np.unique(groups)
    selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    sampled_parts = [np.flatnonzero(groups == group) for group in selected_groups]
    if not sampled_parts:
        return np.array([], dtype=int)
    return np.concatenate(sampled_parts).astype(int)


def _bootstrap_mean_ci(
    values: Sequence[float],
    group_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for a mean; supports optional cluster/group bootstrap."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return None, None
    if group_ids is not None and len(group_ids) != len(values):
        raise ValueError("group_ids length must match values length")

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(len(values), rng, group_ids)
        if len(idx):
            estimates.append(float(np.mean(values[idx])))

    if not estimates:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return _to_python_float(low), _to_python_float(high)


def _bootstrap_statistic_ci(
    x: Sequence[float],
    y: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], Optional[float]],
    group_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 3000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for a statistic of paired x/y observations."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if len(x) < 3:
        return None, None
    if group_ids is not None and len(group_ids) != len(x):
        raise ValueError("group_ids length must match x/y length")

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(len(x), rng, group_ids)
        if len(idx) < 3:
            continue
        estimate = statistic(x[idx], y[idx])
        if estimate is not None and np.isfinite(estimate):
            estimates.append(float(estimate))

    if len(estimates) < max(100, int(0.1 * n_boot)):
        return None, None
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return _to_python_float(low), _to_python_float(high)


def _paired_wilcoxon_greater(differences: Sequence[float]) -> Optional[float]:
    """One-sided Wilcoxon signed-rank p-value for differences > 0."""
    d = np.asarray(differences, dtype=float).reshape(-1)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0.0):
        return 1.0
    try:
        return _to_python_float(
            wilcoxon(d, alternative="greater", zero_method="pratt").pvalue
        )
    except (ValueError, FloatingPointError):
        return None


def _save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def _get_base_model_logits(ensemble: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return logits from every base model, independently of the routing mask."""
    if not hasattr(ensemble, "compute_all_model_logits"):
        raise AttributeError(
            "The ensemble must implement compute_all_model_logits(image), "
            "returning logits for every base model independently of the mask."
        )
    logits = ensemble.compute_all_model_logits(image)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = torch.as_tensor(logits)
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    elif logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(
            "compute_all_model_logits must return [batch, num_models] "
            f"or [batch, num_models, 1], got shape {tuple(logits.shape)}"
        )
    return logits


def _collect_one_sample(
    ensemble: torch.nn.Module,
    dataset: Any,
    index: int,
    device: torch.device,
) -> Tuple[int, np.ndarray, np.ndarray, int]:
    """Collect label, final routing weights, base correctness, ensemble correctness."""
    image, label = dataset[index]
    if not torch.is_tensor(image):
        raise TypeError(f"Dataset image at index {index} is not a torch.Tensor")
    label_int = int(torch.as_tensor(label).item())
    if label_int not in (0, 1):
        raise ValueError(
            f"Expected binary labels 0/1 (0=fake, 1=real), got {label_int} "
            f"at dataset index {index}"
        )

    image = image.unsqueeze(0).to(device, non_blocking=True)
    output, final_weights, _memberships, _selected_outputs = ensemble(
        image, return_details=True
    )

    weights = final_weights.detach().reshape(-1).cpu().numpy().astype(float)
    base_logits = _get_base_model_logits(ensemble, image)
    base_logits = base_logits[0].detach().cpu().numpy().astype(float)

    if len(weights) != len(base_logits):
        raise ValueError(
            f"Number of final weights ({len(weights)}) does not match "
            f"number of base-model logits ({len(base_logits)})"
        )
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(base_logits)):
        raise FloatingPointError(f"Non-finite weights/logits at dataset index {index}")

    base_predictions = (base_logits > 0.0).astype(int)
    base_correct = (base_predictions == label_int).astype(int)

    ensemble_logit = float(output.detach().reshape(-1)[0].cpu().item())
    ensemble_prediction = int(ensemble_logit > 0.0)
    ensemble_correct = int(ensemble_prediction == label_int)

    return label_int, weights, base_correct, ensemble_correct


# ---------------------------------------------------------------------------
# 1) Do the learned routing weights reflect per-sample base-model competence?
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_model_competence(
    ensemble: torch.nn.Module,
    loader: Any,
    device: torch.device,
    model_names: Sequence[str],
    save_dir: str,
    tag: str = "test",
    group_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 5000,
) -> Dict[str, Any]:
    """Evaluate whether HFDE assigns larger weights to correct base models.

    Primary paired comparison:
        weighted_competence[s] = sum_j weight[s,j] * correct[s,j]
        uniform_competence[s] = mean_j correct[s,j]

    The sample-wise difference is tested with one-sided Wilcoxon and a bootstrap
    CI. This is a routing/competence diagnostic, not ensemble accuracy.

    group_ids can be supplied for cluster bootstrap (e.g., video IDs when
    multiple test frames come from the same video).
    """
    ensemble.eval()
    dataset = loader.dataset
    num_models = len(model_names)
    if num_models < 1:
        raise ValueError("model_names must contain at least one model")
    if group_ids is not None and len(group_ids) != len(dataset):
        raise ValueError("group_ids must have one entry per dataset sample")

    all_weights: List[np.ndarray] = []
    all_base_correct: List[np.ndarray] = []
    all_ensemble_correct: List[int] = []

    for i in tqdm(range(len(dataset)), desc=f"Model competence [{tag}]"):
        try:
            _label, weights, base_correct, ensemble_correct = _collect_one_sample(
                ensemble, dataset, i, device
            )
        except Exception as exc:
            # Do not silently drop samples: that can misalign group IDs and labels.
            raise RuntimeError(
                f"Model-competence evaluation failed at dataset index {i}: {exc}"
            ) from exc

        if len(weights) != num_models:
            raise ValueError(
                f"model_names has {num_models} entries, but the model returned "
                f"{len(weights)} weights at sample {i}"
            )
        all_weights.append(weights)
        all_base_correct.append(base_correct)
        all_ensemble_correct.append(ensemble_correct)

    if not all_weights:
        raise RuntimeError("No samples were evaluated")

    weights_np = np.stack(all_weights, axis=0)                  # [S, M]
    correct_np = np.stack(all_base_correct, axis=0).astype(int) # [S, M]
    ensemble_correct_np = np.asarray(all_ensemble_correct, dtype=int)
    n_samples, observed_models = weights_np.shape

    if observed_models != num_models:
        raise ValueError("Observed model count differs from model_names")
    if not np.allclose(weights_np.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(
            "Final routing weights are expected to sum to 1 for each sample. "
            "Check the ensemble's masking and weight-renormalization logic."
        )

    # Expected correctness under learned weights versus uniform weights.
    weighted_competence = np.sum(weights_np * correct_np, axis=1)
    uniform_competence = np.mean(correct_np, axis=1)
    competence_delta = weighted_competence - uniform_competence

    # Which model receives the highest final weight on each sample?
    top1_model_idx = np.argmax(weights_np, axis=1)
    top1_correct = correct_np[np.arange(n_samples), top1_model_idx]
    top1_delta_vs_uniform = top1_correct.astype(float) - uniform_competence

    # Oracle: an upper bound that knows which base models are correct per sample.
    oracle_correct = np.max(correct_np, axis=1)

    delta_ci = _bootstrap_mean_ci(
        competence_delta, group_ids=group_ids, n_boot=n_boot, seed=42
    )
    top1_delta_ci = _bootstrap_mean_ci(
        top1_delta_vs_uniform, group_ids=group_ids, n_boot=n_boot, seed=43
    )

    result: Dict[str, Any] = {
        "split": tag,
        "num_samples": int(n_samples),
        "num_models": int(num_models),
        "ensemble_accuracy": float(ensemble_correct_np.mean()),
        "per_model": {},
        "routing_competence": {
            "mean_weighted_competence": float(weighted_competence.mean()),
            "mean_uniform_competence": float(uniform_competence.mean()),
            "mean_improvement_over_uniform": float(competence_delta.mean()),
            "improvement_95pct_bootstrap_ci": [delta_ci[0], delta_ci[1]],
            "wilcoxon_one_sided_pvalue": _paired_wilcoxon_greater(competence_delta),
            "top1_model_correct_rate": float(top1_correct.mean()),
            "top1_improvement_over_uniform": float(top1_delta_vs_uniform.mean()),
            "top1_improvement_95pct_bootstrap_ci": [top1_delta_ci[0], top1_delta_ci[1]],
            "oracle_accuracy": float(oracle_correct.mean()),
            "ensemble_accuracy": float(ensemble_correct_np.mean()),
        },
        "per_sample": {
            "weights": weights_np.tolist(),
            "base_model_correctness": correct_np.tolist(),
            "ensemble_correctness": ensemble_correct_np.tolist(),
            "weighted_competence": weighted_competence.tolist(),
            "uniform_competence": uniform_competence.tolist(),
            "top1_model_index": top1_model_idx.tolist(),
            "top1_model_correct": top1_correct.tolist(),
        },
    }

    print("\n" + "=" * 100)
    print(f"MODEL-COMPETENCE VALIDATION — {tag.upper()}")
    print("=" * 100)
    print(f"Samples: {n_samples} | Ensemble accuracy: {ensemble_correct_np.mean() * 100:.2f}%")
    print("\nPer-model weight/correctness association:")
    print(f"{'Model':<25} {'Accuracy':>10} {'Mean weight':>13} "
          f"{'Spearman rho':>14} {'p-value':>11} {'W(correct)-W(wrong)':>22} "
          f"{'95% CI':>22}")
    print("-" * 125)

    for j, name in enumerate(model_names):
        w = weights_np[:, j]
        c = correct_np[:, j]
        rho, rho_p = _safe_correlation(w, c, method="spearman")

        if np.any(c == 1) and np.any(c == 0):
            delta_w = float(w[c == 1].mean() - w[c == 0].mean())
            # Bootstrap the conditional mean-weight difference.
            def _conditional_delta(x: np.ndarray, y: np.ndarray) -> Optional[float]:
                correct_mask = y.astype(int) == 1
                wrong_mask = ~correct_mask
                if not np.any(correct_mask) or not np.any(wrong_mask):
                    return None
                return float(x[correct_mask].mean() - x[wrong_mask].mean())

            # Encode correctness in y and bootstrap the pair (weight, correctness).
            ci_w = _bootstrap_statistic_ci(
                w, c,
                lambda x, y: _conditional_delta(x, y),
                group_ids=group_ids,
                n_boot=min(n_boot, 3000),
                seed=100 + j,
            )
        else:
            delta_w, ci_w = None, (None, None)

        result["per_model"][name] = {
            "accuracy": float(c.mean()),
            "mean_final_weight": float(w.mean()),
            "spearman_weight_vs_correctness": rho,
            "spearman_pvalue": rho_p,
            "mean_weight_when_correct": (
                float(w[c == 1].mean()) if np.any(c == 1) else None
            ),
            "mean_weight_when_incorrect": (
                float(w[c == 0].mean()) if np.any(c == 0) else None
            ),
            "weight_difference_correct_minus_incorrect": delta_w,
            "weight_difference_95pct_bootstrap_ci": [ci_w[0], ci_w[1]],
            "times_selected_top1": int(np.sum(top1_model_idx == j)),
            "fraction_selected_top1": float(np.mean(top1_model_idx == j)),
            "accuracy_when_selected_top1": (
                float(correct_np[top1_model_idx == j, j].mean())
                if np.any(top1_model_idx == j) else None
            ),
        }

        ci_text = (
            f"[{ci_w[0]:+.4f}, {ci_w[1]:+.4f}]"
            if ci_w[0] is not None and ci_w[1] is not None else "N/A"
        )
        rho_text = f"{rho:+.4f}" if rho is not None else "N/A"
        p_text = f"{rho_p:.4g}" if rho_p is not None else "N/A"
        dw_text = f"{delta_w:+.4f}" if delta_w is not None else "N/A"
        print(f"{name:<25} {c.mean():>9.2%} {w.mean():>13.4f} "
              f"{rho_text:>14} {p_text:>11} {dw_text:>22} {ci_text:>22}")

    print("\nLearned routing vs uniform weighting:")
    print(f"  Weighted competence : {weighted_competence.mean():.4f}")
    print(f"  Uniform competence  : {uniform_competence.mean():.4f}")
    print(f"  Improvement         : {competence_delta.mean():+.4f}")
    print(f"  95% bootstrap CI    : [{delta_ci[0]}, {delta_ci[1]}]")
    print(f"  Wilcoxon p-value    : {result['routing_competence']['wilcoxon_one_sided_pvalue']}")
    print(f"  Top-1 selected model accuracy: {top1_correct.mean():.2%}")
    print(f"  Oracle upper bound          : {oracle_correct.mean():.2%}")
    print("=" * 100)

    path = os.path.join(save_dir, f"competence_validation_{tag}.json")
    _save_json(result, path)
    print(f"Saved: {path}")
    return result


# ---------------------------------------------------------------------------
# 2) Does the variance-based membership-dispersion score track competence?
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_variance_vs_competence(
    ensemble: torch.nn.Module,
    loader: Any,
    device: torch.device,
    model_names: Sequence[str],
    save_dir: str,
    tag: str = "test",
    group_ids: Optional[Sequence[Any]] = None,
    n_boot: int = 3000,
) -> Dict[str, Any]:
    """Relate the paper's variance score to ensemble/base-model correctness.

    The score exactly matches the ensemble's current variance calculation:
        memberships.var(dim=2, unbiased=False).mean(dim=1)

    Both AUROC directions are reported because high variance may be associated
    with either correctness or error. AUC(error, -variance) tests whether low
    variance is more indicative of error. This function does not assume the
    expected direction in advance.
    """
    ensemble.eval()
    dataset = loader.dataset
    num_models = len(model_names)
    if group_ids is not None and len(group_ids) != len(dataset):
        raise ValueError("group_ids must have one entry per dataset sample")

    variances: List[float] = []
    ensemble_correctness: List[int] = []
    base_correctness: List[np.ndarray] = []
    top1_correctness: List[int] = []

    for i in tqdm(range(len(dataset)), desc=f"Variance/competence [{tag}]"):
        try:
            image, label = dataset[i]
            if not torch.is_tensor(image):
                raise TypeError("Dataset image is not a torch.Tensor")
            label_int = int(torch.as_tensor(label).item())
            if label_int not in (0, 1):
                raise ValueError(f"Expected binary label 0/1, got {label_int}")

            image = image.unsqueeze(0).to(device, non_blocking=True)
            output, final_weights, memberships, _outputs = ensemble(
                image, return_details=True
            )
            if memberships.ndim != 3:
                raise ValueError(
                    "Membership tensor must have shape [batch, models, memberships], "
                    f"got {tuple(memberships.shape)}"
                )

            # Same variance definition as FuzzyHesitantEnsemble._compute_hesitancy.
            h_var = memberships.var(dim=2, unbiased=False).mean(dim=1)
            variances.append(float(h_var.reshape(-1)[0].cpu().item()))

            ensemble_logit = float(output.detach().reshape(-1)[0].cpu().item())
            ensemble_prediction = int(ensemble_logit > 0.0)
            ensemble_correctness.append(int(ensemble_prediction == label_int))

            all_logits = _get_base_model_logits(ensemble, image)[0]
            if len(all_logits) != num_models:
                raise ValueError(
                    f"Expected {num_models} base models, got {len(all_logits)}"
                )
            base_pred = (all_logits.detach().cpu().numpy() > 0.0).astype(int)
            sample_base_correct = (base_pred == label_int).astype(int)
            base_correctness.append(sample_base_correct)
            top_idx = int(torch.argmax(final_weights, dim=1).item())
            top1_correctness.append(int(sample_base_correct[top_idx]))

        except Exception as exc:
            raise RuntimeError(
                f"Variance/competence evaluation failed at dataset index {i}: {exc}"
            ) from exc

    if not variances:
        raise RuntimeError("No samples were evaluated")

    h_var_np = np.asarray(variances, dtype=float)
    ensemble_correct_np = np.asarray(ensemble_correctness, dtype=int)
    base_correct_np = np.stack(base_correctness, axis=0).astype(int)
    mean_base_correct = base_correct_np.mean(axis=1)
    any_base_correct = base_correct_np.max(axis=1)
    all_base_correct = base_correct_np.min(axis=1)

    # Correctness of the highest-weighted model, using the actual final weights
    # returned by the same forward pass (after mask application/renormalization).
    top1_correct_np = np.asarray(top1_correctness, dtype=int)

    ensemble_error_np = 1 - ensemble_correct_np
    results: Dict[str, Any] = {
        "split": tag,
        "num_samples": int(len(h_var_np)),
        "num_models": int(num_models),
        "variance_summary": {
            "mean": float(np.mean(h_var_np)),
            "median": float(np.median(h_var_np)),
            "std": float(np.std(h_var_np)),
            "min": float(np.min(h_var_np)),
            "max": float(np.max(h_var_np)),
        },
        "ensemble_accuracy": float(ensemble_correct_np.mean()),
        "per_model_accuracy": {
            name: float(base_correct_np[:, j].mean())
            for j, name in enumerate(model_names)
        },
        "correlations": {},
        "auroc": {},
        "group_comparisons": {},
    }

    targets = {
        "mean_base_model_correctness": mean_base_correct,
        "ensemble_correctness": ensemble_correct_np,
        "any_base_model_correct": any_base_correct,
        "all_base_models_correct": all_base_correct,
        "top1_weighted_model_correct": top1_correct_np,
    }

    for name, target in targets.items():
        pearson_r, pearson_p = _safe_correlation(h_var_np, target, "pearson")
        spearman_r, spearman_p = _safe_correlation(h_var_np, target, "spearman")

        spearman_ci = _bootstrap_statistic_ci(
            h_var_np,
            target,
            lambda x, y: _safe_correlation(x, y, "spearman")[0],
            group_ids=group_ids,
            n_boot=n_boot,
            seed=200 + len(results["correlations"]),
        )
        results["correlations"][name] = {
            "pearson_r": pearson_r,
            "pearson_pvalue": pearson_p,
            "spearman_r": spearman_r,
            "spearman_pvalue": spearman_p,
            "spearman_95pct_bootstrap_ci": [spearman_ci[0], spearman_ci[1]],
        }

    # For correctness targets, AUC(variance) means higher variance predicts correct.
    for name, target in [
        ("ensemble_correctness", ensemble_correct_np),
        ("any_base_model_correct", any_base_correct),
        ("all_base_models_correct", all_base_correct),
        ("top1_weighted_model_correct", top1_correct_np),
    ]:
        results["auroc"][f"higher_variance_predicts_{name}"] = _safe_auc(target, h_var_np)

    # Direct reviewer-facing error-detection comparison.
    results["auroc"]["ensemble_error_higher_variance_score"] = _safe_auc(
        ensemble_error_np, h_var_np
    )
    results["auroc"]["ensemble_error_lower_variance_score"] = _safe_auc(
        ensemble_error_np, -h_var_np
    )

    for name, target in [
        ("ensemble_correctness", ensemble_correct_np),
        ("mean_base_model_correctness_at_least_one_half", (mean_base_correct >= 0.5).astype(int)),
        ("any_base_model_correct", any_base_correct),
        ("all_base_models_correct", all_base_correct),
        ("top1_weighted_model_correct", top1_correct_np),
    ]:
        if np.any(target == 1) and np.any(target == 0):
            var_correct = float(h_var_np[target == 1].mean())
            var_incorrect = float(h_var_np[target == 0].mean())
            delta = var_correct - var_incorrect
            results["group_comparisons"][name] = {
                "mean_variance_correct": var_correct,
                "mean_variance_incorrect": var_incorrect,
                "difference_correct_minus_incorrect": delta,
            }
        else:
            results["group_comparisons"][name] = None

    results["per_model_variance_competence"] = {}
    for j, name in enumerate(model_names):
        r, p = _safe_correlation(h_var_np, base_correct_np[:, j], "spearman")
        results["per_model_variance_competence"][name] = {
            "base_model_accuracy": float(base_correct_np[:, j].mean()),
            "spearman_variance_vs_correctness": r,
            "spearman_pvalue": p,
        }

    key = results["correlations"]["mean_base_model_correctness"]
    print("\n" + "=" * 100)
    print(f"VARIANCE-BASED HESITANCY vs MODEL COMPETENCE — {tag.upper()}")
    print("=" * 100)
    print(f"Samples: {len(h_var_np)} | Ensemble accuracy: {ensemble_correct_np.mean() * 100:.2f}%")
    print(f"Mean variance: {h_var_np.mean():.8f} | Median: {np.median(h_var_np):.8f}")
    print("\nVariance vs mean base-model correctness:")
    print(f"  Spearman rho : {key['spearman_r']}")
    print(f"  p-value      : {key['spearman_pvalue']}")
    print(f"  95% bootstrap CI: {key['spearman_95pct_bootstrap_ci']}")
    print("\nEnsemble error detection:")
    print("  AUROC using +variance :",
          results["auroc"]["ensemble_error_higher_variance_score"])
    print("  AUROC using -variance :",
          results["auroc"]["ensemble_error_lower_variance_score"])
    print("\nPer-model association:")
    for name, item in results["per_model_variance_competence"].items():
        print(f"  {name}: accuracy={item['base_model_accuracy']:.3f}, "
              f"Spearman={item['spearman_variance_vs_correctness']}, "
              f"p={item['spearman_pvalue']}")
    print("=" * 100)

    path = os.path.join(save_dir, f"variance_competence_validation_{tag}.json")
    _save_json(results, path)
    print(f"Saved: {path}")
    return results
