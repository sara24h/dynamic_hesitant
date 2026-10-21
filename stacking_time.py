import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from old_dataset_utils import create_dataloaders
from visualization_utils import generate_visualizations

from efficiency_utils import (
    Stopwatch,
    benchmark_efficiency,
    summarize_training_times,
)


# ================== UTILITY ==================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}',  torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')


# ================== STACKING ENSEMBLE ==================
class StackingMetaLearner(nn.Module):
    def __init__(self, num_models: int):
        super().__init__()
        self.linear = nn.Linear(num_models, 1)
    def forward(self, x):
        return self.linear(x)


class StackingEnsemble(nn.Module):
    def __init__(self, models, means, stds, freeze_models=True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.meta_learner = StackingMetaLearner(self.num_models)
        if freeze_models:
            for m in self.models:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x, return_details=False):
        logits = []
        for i in range(self.num_models):
            xn = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](xn)
                if isinstance(out, (tuple, list)): out = out[0]
            logits.append(out)
        stacked = torch.cat(logits, dim=1)
        final = self.meta_learner(stacked)
        bs = x.size(0)
        dummy_w = torch.ones(bs, self.num_models, device=x.device) / self.num_models
        if return_details:
            dummy_m = torch.zeros(bs, self.num_models, 3, device=x.device)
            return final, dummy_w, dummy_m, stacked
        return final, dummy_w


class _StackingBenchmarkAdapter(nn.Module):
    def __init__(self, ensemble: StackingEnsemble):
        super().__init__()
        self.ensemble = ensemble
        self.models = ensemble.models
        self.normalizations = ensemble.normalizations
        self.num_models = ensemble.num_models

    def eval(self):
        self.ensemble.eval(); return self

    @torch.no_grad()
    def hesitant_fuzzy(self, x):
        return torch.full((x.size(0), self.num_models),
                          1.0 / self.num_models, device=x.device)

    def forward(self, x):
        out, w = self.ensemble(x)
        return out, w


# ================== MODEL LOADING ==================
def load_pruned_models(paths, device, is_main):
    from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    models = []
    if is_main: print(f"Loading {len(paths)} pruned models...")
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            if is_main: print(f" [WARNING] Missing: {p}")
            continue
        if is_main: print(f" [{i+1}/{len(paths)}] {os.path.basename(p)}")
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            m = ResNet_50_pruned_hardfakevsreal(masks=ck['masks'])
            m.load_state_dict(ck['model_state_dict'])
            m = m.to(device).eval()
            if is_main:
                print(f" → Params: {sum(x.numel() for x in m.parameters()):,}")
            models.append(m)
        except Exception as e:
            if is_main: print(f" [ERROR] {e}")
    if not models: raise ValueError("No models loaded")
    return models


def _reduce_correct_total(correct, total, device):
    c = torch.tensor(correct, dtype=torch.long, device=device)
    t = torch.tensor(total,   dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return c.item(), t.item()


@torch.no_grad()
def evaluate_single_model_ddp(model, loader, device, name, mean, std, is_main):
    model.eval()
    norm = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    seen = 0
    for images, labels in tqdm(loader, desc=f"Eval {name}", disable=not is_main):
        images = norm(images.to(device), 0)
        labels = labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)): out = out[0]
        pred = (out.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        seen += labels.size(0)

    correct, seen = _reduce_correct_total(correct, seen, device)
    acc = 100. * correct / seen if seen > 0 else 0.0
    if is_main:
        print(f" {name}: {acc:.2f}%  (samples seen: {seen})")
    return acc


@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device):
    model.eval()
    correct = 0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        correct += pred.eq(labels.long()).sum().item()
        seen += labels.size(0)

    correct, seen = _reduce_correct_total(correct, seen, device)
    return 100. * correct / seen if seen > 0 else 0.0


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    correct = 0
    seen = 0
    ws = dist.get_world_size() if dist.is_initialized() else 1

    all_true, all_score, all_pred = [], [], []

    if is_main:
        print(f"\nEvaluating {name} set (Stacking - LR)...")

    for images, labels in tqdm(loader, desc=f"Eval {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        out, _ = model(images, return_details=False)
        probs = torch.sigmoid(out.squeeze(1))
        pred_int = (probs > 0.5).long()

        all_true.extend(labels.cpu().numpy().tolist())
        all_score.extend(probs.cpu().numpy().tolist())
        all_pred.extend(pred_int.cpu().numpy().tolist())

        correct += pred_int.eq(labels.long()).sum().item()
        seen    += labels.size(0)

    correct, seen = _reduce_correct_total(correct, seen, device)

    if ws > 1:
        gt = [None for _ in range(ws)]; gs = [None for _ in range(ws)]; gp = [None for _ in range(ws)]
        dist.all_gather_object(gt, all_true)
        dist.all_gather_object(gs, all_score)
        dist.all_gather_object(gp, all_pred)
        if is_main:
            all_true  = [i for s in gt for i in s]
            all_score = [i for s in gs for i in s]
            all_pred  = [i for s in gp for i in s]

    if not is_main:
        return 0.0, [0.0]*len(model_names), 0.0, ([], [], [])

    # برای JSON: با shuffle=False و DistributedSampler، ترتیب concat برابر
    # با ترتیب دیتاست است؛ پس truncate به len(dataset) درست است.
    N = len(loader.dataset)
    all_true  = all_true[:N]
    all_score = all_score[:N]
    all_pred  = all_pred[:N]

    acc = 100. * correct / seen if seen > 0 else 0.0

    meta = model.meta_learner
    w = meta.linear.weight.data.cpu().squeeze().numpy()
    b = meta.linear.bias.data.cpu().item()

    print(f"\n{'='*70}")
    print(f"{name.upper()} SET RESULTS (Stacking - LR)")
    print(f"{'='*70}")
    print(f" → Accuracy: {acc:.3f}%  (evaluated: {seen})")
    print(f"\nMeta-Learner (LR):")
    print(f"  Bias: {b:+.4f}")
    for i, (wi, mn) in enumerate(zip(w, model_names)):
        print(f"    {i+1:2d}. {mn:<25}: {wi:+.4f}")
    print(f"{'='*70}")
    return acc, w.tolist(), b, (all_true, all_score, all_pred)


# ================== TRAINING ==================
def train_stacking(ensemble_model, train_loader, val_loader, num_epochs, lr,
                   device, save_dir, is_main, model_names):
    os.makedirs(save_dir, exist_ok=True)
    actual = ensemble_model.module if hasattr(ensemble_model, 'module') else ensemble_model
    meta = actual.meta_learner

    optimizer = torch.optim.AdamW(meta.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    epoch_times, train_times, val_times = [], [], []
    total_sw = Stopwatch(device).start()

    if is_main:
        print("="*70); print("Training Stacking LR"); print("="*70)
        print(f"Trainable: {sum(p.numel() for p in meta.parameters()):,}")
        print(f"Epochs: {num_epochs} | LR: {lr} | Models: {len(model_names)}\n")

    for ep in range(num_epochs):
        epoch_sw = Stopwatch(device).start()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(ep)

        ensemble_model.train()
        for m in actual.models:
            m.eval()

        train_loss_sum = 0.0
        train_correct = 0
        train_seen = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {ep+1}/{num_epochs}',
                                   disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            out, _ = ensemble_model(images)
            loss = criterion(out.squeeze(1), labels)
            loss.backward(); optimizer.step()

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            pred = (out.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_seen += bs

        t_train = epoch_sw.stop()

        # ---- DDP-safe aggregate ----
        tl_t = torch.tensor(train_loss_sum, device=device)
        tc_t = torch.tensor(train_correct,  device=device)
        ts_t = torch.tensor(train_seen,     device=device)
        if dist.is_initialized():
            dist.all_reduce(tl_t); dist.all_reduce(tc_t); dist.all_reduce(ts_t)
        train_loss = tl_t.item() / max(ts_t.item(), 1)
        train_acc  = 100. * tc_t.item() / max(ts_t.item(), 1)

        val_sw = Stopwatch(device).start()
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        t_val = val_sw.stop()

        # t_epoch = t_train + t_val (syncs اضافی Stopwatch را حذف می‌کنیم)
        t_epoch = t_train + t_val
        epoch_times.append(t_epoch)
        train_times.append(t_train)
        val_times.append(t_val)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if is_main:
            print(f"\n{'='*70}\nEpoch {ep+1}/{num_epochs}\n{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Time: epoch {t_epoch:.1f}s (train {t_train:.1f}s | val {t_val:.1f}s)")

        if is_main and val_acc > best_val:
            best_val = val_acc
            torch.save({
                'epoch': ep + 1,
                'meta_state_dict': meta.state_dict(),
                'val_acc': val_acc,
                'history': history,
            }, os.path.join(save_dir, 'best_stacking_lr.pt'))
            print(f"\n✓ Best saved → {val_acc:.2f}%")

    total_time = total_sw.stop()
    timing = summarize_training_times(epoch_times, train_times, val_times, total_time)

    if is_main:
        print(f"\n{'='*70}\nTRAINING TIME SUMMARY\n{'='*70}")
        print(f"Total       : {timing['total_training_time_s']:.1f}s ({timing['total_training_time_min']:.2f} min)")
        print(f"Mean epoch  : {timing['mean_epoch_time_s']:.1f}s")
        print(f"Mean (excl 1st): {timing['mean_epoch_time_excl_first_s']:.1f}s")
        print(f"Mean train  : {timing['mean_train_phase_s']:.1f}s")
        print(f"Mean val    : {timing['mean_val_phase_s']:.1f}s")
        print(f"{'='*70}\n")

        try:
            plt.figure(figsize=(14, 5))
            r = range(1, num_epochs + 1)
            plt.subplot(1, 2, 1)
            plt.plot(r, history['train_loss'], label='train', color='blue', lw=2)
            plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.grid(True, ls='--', alpha=0.7); plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(r, history['train_acc'], label='train', color='blue', lw=2)
            plt.plot(r, history['val_acc'], label='val', color='orange', lw=2)
            plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc (%)')
            plt.grid(True, ls='--', alpha=0.7); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Curves saved.")
        except Exception as e:
            print(f"[WARN] Curves failed: {e}")

    return best_val, history, timing


# ================== DDP SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        ws   = int(os.environ['WORLD_SIZE'])
        lr_  = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(lr_)
        return torch.device(f'cuda:{lr_}'), lr_, rank, ws
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ================== METRICS ==================
def compute_real_fake_metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    rp, rr, rf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None, zero_division=0)
    fp_, fr, ff, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0], average=None, zero_division=0)
    rp, rr, rf = rp[0], rr[0], rf[0]
    fp_, fr, ff = fp_[0], fr[0], ff[0]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    return {
        'real': {'precision': float(rp), 'recall': float(rr), 'f1': float(rf),
                 'specificity': float(TN/(TN+FP)) if (TN+FP) else 0.0,
                 'support': int(np.sum(y_true == 1))},
        'fake': {'precision': float(fp_), 'recall': float(fr), 'f1': float(ff),
                 'specificity': float(TP/(TP+FN)) if (TP+FN) else 0.0,
                 'support': int(np.sum(y_true == 0))},
        'macro_f1': float((rf + ff) / 2),
        'confusion_matrix': {'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)},
    }


def print_real_fake_metrics(m, title="REAL/FAKE"):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"{'Class':<10}{'Prec':<12}{'Rec':<12}{'F1':<12}{'Spec':<12}{'Support':<10}")
    print('-'*70)
    for c in ('real', 'fake'):
        x = m[c]
        print(f"{c.upper():<10}{x['precision']:<12.4f}{x['recall']:<12.4f}"
              f"{x['f1']:<12.4f}{x['specificity']:<12.4f}{x['support']:<10}")
    print('-'*70)
    print(f"Macro F1: {m['macro_f1']:.4f}")
    print(f"CM: TN={m['confusion_matrix']['TN']}, FP={m['confusion_matrix']['FP']}, "
          f"FN={m['confusion_matrix']['FN']}, TP={m['confusion_matrix']['TP']}")
    print('='*70)


def save_predictions_json(y_true, y_score, y_pred, path, seed, ds):
    out = {"metadata": {"seed": seed, "dataset": ds, "num_samples": len(y_true),
                        "positive_count": sum(y_true),
                        "negative_count": len(y_true) - sum(y_true)},
           "y_true": y_true, "y_score": y_score, "y_pred": y_pred}
    with open(path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"Predictions JSON: {path}")


def save_accuracy_log_from_results(y_true, y_pred, path, model_name):
    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = 100. * correct / total if total else 0.0
    TP = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    TN = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    FP = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    FN = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    lines = ["="*70, f"Model: {model_name}", "="*70,
             f"Accuracy: {acc:.3f}%", f"Total: {total}",
             f"Correct: {correct}", f"Incorrect: {total-correct}",
             "", "Confusion Matrix:",
             f"                 Pred Fake  Pred Real",
             f"    Actual Fake   {TN:<10} {FP:<10}",
             f"    Actual Real   {FN:<10} {TP:<10}",
             "", "Sample-by-sample:",
             f"{'ID':<8}{'True':<8}{'Pred':<8}{'Status':<10}", '-'*40]
    for i, (a, b) in enumerate(zip(y_true, y_pred)):
        lines.append(f"{i+1:<8}{a:<8}{b:<8}{'Correct' if a==b else 'Wrong':<10}")
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Log saved: {path}")


# ================== CORRECTED LATENCY / THROUGHPUT / COMPUTE ==================
# FIX خلاصه:
#   • داده برای latency و throughput یک‌بار روی GPU preload می‌شود (H2D از حلقه حذف شد).
#   • Latency با cuda.Event (زمان خالص GPU) به‌جای perf_counter.
#   • bs=1 هم warmup می‌شود + clock ramp-up.
#   • throughput و computational time دیگر duplicate نیستند:
#       - throughput: روی batch‌های preloaded (با bs=loader.batch_size)
#       - computational time: یک pass کامل روی test loader، H2D هر batch حذف می‌شود
#                              (سازگار با measure_full_pass در efficiency_utils).
@torch.no_grad()
def measure_latency_throughput(model, loader, device, is_main=True, num_runs=10,
                               n_lat=100, bs_tp=None, warmup=30, ramp_up_s=2.0):
    model.eval()
    bs_tp = bs_tp or (loader.batch_size or 16)

    # ---- latency inputs: preload on GPU ----
    n_lat = min(n_lat, len(loader.dataset))
    lat_x = torch.stack([loader.dataset[i][0] for i in range(n_lat)]).to(device)

    # ---- throughput inputs: preload on GPU (چند batch کافی است) ----
    tp_x = []
    for imgs, _ in loader:
        tp_x.append(imgs.to(device))
        if len(tp_x) >= 20:
            break

    # ---- warmup: هر دو batch size + clock ramp-up ----
    for x in (lat_x[:1], tp_x[0]):
        for _ in range(warmup):
            _ = model(x)
    _sync(device)
    if device.type == 'cuda' and ramp_up_s > 0:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ramp_up_s:
            _ = model(tp_x[0]); _sync(device)

    # ---- latency (CUDA events, GPU-only) ----
    lat_ms = []
    for i in range(n_lat):
        x = lat_x[i:i+1]
        if device.type == 'cuda':
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _ = model(x); e.record(); _sync(device)
            lat_ms.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); _ = model(x)
            lat_ms.append((time.perf_counter() - t0) * 1000.)
    lat_mean = float(np.mean(lat_ms))
    lat_std  = float(np.std(lat_ms, ddof=1)) if len(lat_ms) > 1 else 0.0

    # ---- throughput (preloaded batches) ----
    n_per_run = sum(x.size(0) for x in tp_x)
    tp_runs = []
    for _ in range(num_runs):
        _sync(device); t0 = time.perf_counter()
        for x in tp_x:
            _ = model(x)
        _sync(device)
        tp_runs.append(n_per_run / (time.perf_counter() - t0))
    tp_mean = float(np.mean(tp_runs))
    tp_std  = float(np.std(tp_runs, ddof=1)) if len(tp_runs) > 1 else 0.0

    # ---- full-pass compute time (H2D excluded, on whole test set) ----
    ct_runs, n_full = [], 0
    for _ in range(num_runs):
        compute_s = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _sync(device)              # H2D done — not counted below
            t0 = time.perf_counter()
            _ = model(imgs)
            _sync(device)
            compute_s += time.perf_counter() - t0
            n_full = len(loader.dataset)
        ct_runs.append(compute_s)
    ct_mean = float(np.mean(ct_runs))
    ct_std  = float(np.std(ct_runs, ddof=1)) if len(ct_runs) > 1 else 0.0

    if is_main:
        print("="*70)
        print(f"Latency (bs=1)          : {lat_mean:.2f} ± {lat_std:.2f} ms / image  (cuda.Event)")
        print(f"Throughput (bs={bs_tp})".ljust(24) +
              f": {tp_mean:.1f} ± {tp_std:.1f} images/s  (mean ± std of {num_runs} runs)")
        print(f"Computational time      : {ct_mean:.2f} ± {ct_std:.2f} s "
              f"(model inference on the whole test set, {n_full} images, mean of {num_runs} passes; "
              f"H2D excluded)")
        print("="*70)

    return {
        'num_test_images': int(n_full),
        'num_runs': int(num_runs),
        'latency_bs1_ms_per_image':        {'mean': lat_mean, 'std': lat_std},
        'throughput_bs{}_images_per_sec'.format(bs_tp): {'mean': tp_mean, 'std': tp_std},
        'computational_time_sec':          {'mean': ct_mean, 'std': ct_std},
    }


# ================== MAIN ==================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_grad_cam_samples', type=int, default=5)
    p.add_argument('--num_lime_samples', type=int, default=5)
    p.add_argument('--dataset', type=str, required=True,
                   choices=['wild','deepfake_lab','hard_fake_real','custom_genai',
                            'custom_genai_v2','uadfV','real_fake_dataset'])
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--model_paths', type=str, nargs='+', required=True)
    p.add_argument('--model_names', type=str, nargs='+', required=True)
    p.add_argument('--save_dir', type=str, default='./output')
    p.add_argument('--seed', type=int, nargs='+', required=True)
    args = p.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("model_names و model_paths باید هم‌طول باشند")

    for current_seed in args.seed:
        print(f"\n{'#'*80}\n# SEED: {current_seed}\n{'#'*80}\n")
        set_seed(current_seed)
        device, local_rank, rank, world_size = setup_distributed()
        is_main = rank == 0

        try:
            pipeline_sw = Stopwatch(device).start()
            save_dir = os.path.join(args.save_dir, f'seed_{current_seed}')

            if is_main:
                print("="*70); print("STACKING ENSEMBLE (LR)"); print("="*70)
                print(f"GPU(s): {world_size} | Seed: {current_seed} | Dataset: {args.dataset}")
                print(f"Output: {save_dir}\n")

            MEANS = [(0.5207,0.4258,0.3806),(0.4460,0.3622,0.3416),(0.4668,0.3816,0.3414)]
            STDS  = [(0.2490,0.2239,0.2212),(0.2057,0.1849,0.1761),(0.2410,0.2161,0.2081)]
            MEANS = MEANS[:len(args.model_paths)]
            STDS  = STDS[:len(args.model_paths)]

            base_models = load_pruned_models(args.model_paths, device, is_main)
            model_names = args.model_names[:len(base_models)]

            ensemble = StackingEnsemble(base_models, MEANS, STDS, freeze_models=True).to(device)
            if world_size > 1:
                ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

            if is_main:
                trainable = sum(q.numel() for q in ensemble.parameters() if q.requires_grad)
                total     = sum(q.numel() for q in ensemble.parameters())
                print(f"Params: total={total:,} | trainable={trainable:,} | frozen={total-trainable:,}\n")

            train_loader, val_loader, test_loader = create_dataloaders(
                args.data_dir, args.batch_size, dataset_type=args.dataset,
                is_distributed=(world_size > 1), seed=current_seed, is_main=is_main)

            if is_main:
                print("\n" + "="*70); print("BASE MODELS (test)"); print("="*70)
            ind_accs, ind_times = [], []
            for i, m in enumerate(base_models):
                sw = Stopwatch(device).start()
                a = evaluate_single_model_ddp(m, test_loader, device,
                                             f"{model_names[i]}", MEANS[i], STDS[i], is_main)
                t = sw.stop()
                ind_accs.append(a); ind_times.append(t)
                if is_main:
                    print(f"  ⏱ {t:.1f}s")

            best_single = max(ind_accs)
            best_idx = ind_accs.index(best_single)
            if is_main:
                print(f"\nBest: {model_names[best_idx]} → {best_single:.2f}%\n")

            # ---------- Training ----------
            best_val, history, train_timing = train_stacking(
                ensemble, train_loader, val_loader,
                args.epochs, args.lr, device, save_dir, is_main, model_names)

            ck_path = os.path.join(save_dir, 'best_stacking_lr.pt')
            if os.path.exists(ck_path):
                ck = torch.load(ck_path, map_location=device, weights_only=False)
                meta_m = ensemble.module.meta_learner if hasattr(ensemble, 'module') else ensemble.meta_learner
                meta_m.load_state_dict(ck['meta_state_dict'])
                if is_main: print("Best meta-learner loaded.\n")

            ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble

            # ---------- Final test eval ----------
            eval_sw = Stopwatch(device).start()
            acc, w, b, pred_data = evaluate_ensemble_final_ddp(
                ensemble_module, test_loader, device, "Test", model_names, is_main)
            final_eval_time = eval_sw.stop()

            if is_main:
                print(f"\n⏱ Final eval: {final_eval_time:.1f}s")
                print("\n" + "="*70); print("FINAL COMPARISON"); print("="*70)
                print(f"Best Single: {best_single:.2f}%")
                print(f"Stacking   : {acc:.2f}%")
                print(f"Delta      : {acc - best_single:+.2f}%")
                print("="*70)

                yt, ys, yp = pred_data
                rf = compute_real_fake_metrics(yt, yp)
                print_real_fake_metrics(rf, title="STACKING — REAL/FAKE")

                # ---------- Efficiency ----------
                adapter = _StackingBenchmarkAdapter(ensemble_module).to(device)
                bench_sw = Stopwatch(device).start()
                try:
                    perf = benchmark_efficiency(
                        adapter, test_loader, device, model_names,
                        latency_samples=min(300, len(test_loader.dataset)),
                        verbose=is_main)
                except Exception as e:
                    print(f"[WARN] benchmark_efficiency failed: {e}")
                    perf = {}
                bench_time = bench_sw.stop()

                fmt_sw = Stopwatch(device).start()
                try:
                    fmt = measure_latency_throughput(
                        adapter, test_loader, device, is_main=is_main,
                        num_runs=10, n_lat=100, bs_tp=args.batch_size)
                    if not isinstance(perf, dict): perf = {}
                    perf['formatted_summary'] = fmt
                except Exception as e:
                    print(f"[WARN] formatted summary failed: {e}")
                fmt_time = fmt_sw.stop()

                # ---------- Save everything ----------
                save_predictions_json(yt, ys, yp,
                                      os.path.join(save_dir, 'test_predictions.json'),
                                      current_seed, args.dataset)
                save_accuracy_log_from_results(yt, yp,
                                               os.path.join(save_dir, 'prediction_log_stacking.txt'),
                                               "Stacking Ensemble")

                total_pipe = pipeline_sw.stop()
                print("\n" + "="*70); print("TIME SUMMARY"); print("="*70)
                print(f"  Training             : {train_timing['total_training_time_s']:.1f}s "
                      f"(mean epoch {train_timing['mean_epoch_time_s']:.1f}s)")
                print(f"  Individual evals     : {sum(ind_times):.1f}s")
                print(f"  benchmark_efficiency : {bench_time:.1f}s")
                print(f"  formatted summary    : {fmt_time:.1f}s")
                print(f"  Final eval           : {final_eval_time:.1f}s")
                print(f"  Total pipeline       : {total_pipe/60:.2f} min ({total_pipe:.1f}s)")
                print("="*70)

                final_results = {
                    'seed': current_seed, 'method': 'Stacking_LR',
                    'best_single_model': {'name': model_names[best_idx], 'accuracy': float(best_single)},
                    'ensemble': {'test_accuracy': float(acc),
                                 'learned_weights': {n: float(v) for n, v in zip(model_names, w)},
                                 'learned_bias': float(b)},
                    'improvement': float(acc - best_single),
                    'training_history': history,
                    'real_fake_metrics': rf,
                    'performance_metrics': perf,
                    'timing': {
                        'training_time': train_timing,
                        'individual_eval_times_s': {n: float(t) for n, t in zip(model_names, ind_times)},
                        'benchmark_time_s': float(bench_time),
                        'formatted_summary_time_s': float(fmt_time),
                        'final_eval_time_s': float(final_eval_time),
                        'total_pipeline_time_s': float(total_pipe),
                    },
                }
                with open(os.path.join(save_dir, 'final_results_stacking_lr.json'), 'w') as f:
                    json.dump(final_results, f, indent=4)
                print(f"Results: {os.path.join(save_dir, 'final_results_stacking_lr.json')}")

                torch.save({
                    'ensemble_state_dict': ensemble_module.state_dict(),
                    'meta_learner_state_dict': ensemble_module.meta_learner.state_dict(),
                    'test_accuracy': acc, 'model_names': model_names,
                    'means': MEANS, 'stds': STDS, 'seed': current_seed,
                }, os.path.join(save_dir, 'final_stacking_lr_model.pt'))
                print(f"Model: {os.path.join(save_dir, 'final_stacking_lr_model.pt')}")

                # Best single log
                single_model = base_models[best_idx].eval()
                norm = MultiModelNormalization([MEANS[best_idx]], [STDS[best_idx]]).to(device)
                sy, sp = [], []
                for imgs, labels in tqdm(test_loader, desc="Best-single log"):
                    imgs = norm(imgs.to(device), 0)
                    with torch.no_grad():
                        o = single_model(imgs)
                        if isinstance(o, (tuple, list)): o = o[0]
                        pr = (o.squeeze(1) > 0).long()
                    sy.extend(labels.cpu().numpy().tolist())
                    sp.extend(pr.cpu().numpy().tolist())
                N = len(test_loader.dataset)
                save_accuracy_log_from_results(sy[:N], sp[:N],
                                               os.path.join(save_dir, 'prediction_log_best_single.txt'),
                                               model_names[best_idx])

                # Visualizations
                try:
                    generate_visualizations(
                        ensemble_module, test_loader, device,
                        os.path.join(save_dir, 'visualizations'), model_names,
                        args.num_grad_cam_samples, args.num_lime_samples,
                        args.dataset, is_main)
                except Exception as e:
                    print(f"[WARN] visualizations failed: {e}")
        finally:
            if dist.is_initialized():
                dist.barrier()        
            cleanup_distributed()


if __name__ == "__main__":
    main()
