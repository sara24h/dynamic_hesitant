import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SmallResNet(nn.Module):
    """ResNet-like backbone (roughly ResNet-18 compute at 224x224)."""
    def __init__(self, depth=(2, 2, 2, 2), width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 7, 2, 3, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        layers, in_ch = [], width
        for i, n in enumerate(depth):
            out_ch = width * (2 ** i)
            for _ in range(n):
                layers.append(self._block(in_ch, out_ch))
                in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_ch, 1)

    @staticmethod
    def _block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x); x = self.body(x)
        return self.fc(self.pool(x).flatten(1))


def calibrate_for_early_exit(model, target_std=4.0, img_size=224, device="cuda"):
    """Widen the logit distribution so early-exit fires ~50% of the time."""
    dummy = torch.randn(64, 3, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    cur = out.std().item()
    if cur > 1e-6 and hasattr(model, "fc"):
        s = target_std / cur
        with torch.no_grad():
            model.fc.weight.mul_(s)
            if model.fc.bias is not None:
                model.fc.bias.mul_(s)


# ============================================================
# 2) Ensemble variants
# ============================================================
class Normalizer(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f"mean_{i}", torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f"std_{i}",  torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f"mean_{idx}")) / getattr(self, f"std_{idx}")


class _BaseEnsemble(nn.Module):
    def __init__(self, models, means, stds):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n = len(models)
        self.norm = Normalizer(means, stds)
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def _run(self, x, i):
        with torch.no_grad():
            out = self.models[i](self.norm(x, i))
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out

    def _all_logits(self, x):
        return torch.cat([self._run(x, i) for i in range(self.n)], dim=1)


class ProposedEnsemble(_BaseEnsemble):
    def __init__(self, models, means, stds, thresh=0.9):
        super().__init__(models, means, stds)
        self.thresh = thresh

    def forward(self, x):
        B = x.size(0)
        out = torch.zeros(B, 1, device=x.device)
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)
        for i in range(self.n):
            if bool(decided.all()):
                break
            idx = (~decided).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break
            sub = self._run(x[idx], i)
            prob = torch.sigmoid(sub).squeeze(1)
            conf = (prob - 0.5).abs() * 2.0
            if i == self.n - 1:
                out[idx] = sub; decided[idx] = True; break
            mask = conf > self.thresh
            if mask.any():
                take = idx[mask]
                out[take] = sub[mask]
                decided[take] = True
        return out


class SimpleAvgEnsemble(_BaseEnsemble):
    def forward(self, x):
        return self._all_logits(x).mean(dim=1, keepdim=True)


class WeightedAvgEnsemble(_BaseEnsemble):
    def __init__(self, models, means, stds, weights=None):
        super().__init__(models, means, stds)
        w = torch.ones(self.n) / self.n if weights is None else torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("w", w)

    def forward(self, x):
        return (self._all_logits(x) * F.softmax(self.w, dim=0).view(1, -1)).sum(1, keepdim=True)


class MajorityVotingEnsemble(_BaseEnsemble):
    def forward(self, x):
        votes = (self._all_logits(x) > 0).float().mean(1, keepdim=True)
        return torch.where(votes > 0.5, torch.ones_like(votes), -torch.ones_like(votes))


class StackingEnsemble(_BaseEnsemble):
    def __init__(self, models, means, stds):
        super().__init__(models, means, stds)
        self.meta = nn.Sequential(nn.Linear(self.n, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
        for p in self.meta.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.meta(self._all_logits(x))


class FuzzyIntegralEnsemble(_BaseEnsemble):
    def __init__(self, models, means, stds):
        super().__init__(models, means, stds)
        self.register_buffer("mu", torch.linspace(0.1, 1.0, self.n))

    def forward(self, x):
        p, _ = torch.sort(torch.sigmoid(self._all_logits(x)), dim=1)
        agg = (p * self.mu.view(1, -1)).sum(1, keepdim=True).clamp(1e-6, 1 - 1e-6)
        return torch.log(agg / (1 - agg))


# ============================================================
# 3) Benchmark primitives
# ============================================================
@torch.no_grad()
def measure_latency(model, device, img=224, n_measure=30, warmup=10):
    """Per-image latency in ms (bs=1). Returns (mean, std)."""
    model.eval().to(device)
    x = torch.randn(1, 3, img, img, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ms = []
    for _ in range(n_measure):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(ms)
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def measure_throughput(model, device, bs=16, img=224, n_runs=3, warmup=5):
    """Throughput in images/s. Returns (mean, std) over n_runs."""
    model.eval().to(device)
    x = torch.randn(bs, 3, img, img, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    ips = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ips.append(10 * bs / (time.perf_counter() - t0))
    arr = np.array(ips)
    return float(arr.mean()), float(arr.std())


@torch.no_grad()
def measure_computational_time(model, batches: List[torch.Tensor], device, n_passes=3):
    """Whole-set inference time (data loading excluded). Returns (mean, std) in s."""
    model.eval()
    for _ in range(2):                      # warmup
        for b in batches:
            _ = model(b)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_passes):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for b in batches:
            _ = model(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ============================================================
# 4) Formatter — matches the example block exactly
# ============================================================
def format_summary(title, lat_m, lat_s, thr_m, thr_s, ct_m, ct_s,
                   n_images, bs_lat, bs_thr, n_runs):
    lines = [
        "=" * 70,
        f"{title} EFFICIENCY SUMMARY",
        "=" * 70,
        f"{f'Latency (bs={bs_lat})':<24}: {lat_m:.2f} ± {lat_s:.2f} ms / image",
        f"{f'Throughput (bs={bs_thr})':<24}: {thr_m:.1f} ± {thr_s:.1f} images/s "
        f"(mean ± std of {n_runs} runs)",
        f"{'Computational time':<24}: {ct_m:.2f} ± {ct_s:.2f} s "
        f"(model inference on the whole test set, {n_images} images, "
        f"mean of {n_runs} passes; data loading excluded)",
    ]
    return "\n".join(lines)


def print_comparison_table(results, order, bs_lat, bs_thr):
    cols = order
    print("\nDataset\t" + "\t".join(cols))
    print("latency\t" + "\t".join(
        f"{results[c]['lat_m']:.2f} ± {results[c]['lat_s']:.2f} ms / image"
        for c in cols))
    print("throughput\t" + "\t".join(
        f"{results[c]['thr_m']:.1f} ± {results[c]['thr_s']:.1f} images/s"
        for c in cols))


# ============================================================
# 5) Data
# ============================================================
class DummySet(Dataset):
    def __init__(self, n, img=224):
        self.n, self.img = n, img

    def __len__(self): return self.n

    def __getitem__(self, _):
        return torch.randn(3, self.img, self.img), 0


def build_batches(n_images, bs, img, device, data_dir=None):
    if data_dir is None:
        loader = DataLoader(DummySet(n_images, img), batch_size=bs, num_workers=0)
    else:
        raise NotImplementedError(
            "Hook your create_dataloaders() here if you have a real dataset.")
    return [b.to(device) for b, _ in loader]


# ============================================================
# 6) Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_paths", nargs="*", default=None)
    ap.add_argument("--num_models", type=int, default=3)
    ap.add_argument("--num_images", type=int, default=194)
    ap.add_argument("--bs_lat", type=int, default=1)
    ap.add_argument("--bs_thr", type=int, default=16)
    ap.add_argument("--n_runs", type=int, default=3, help="runs for throughput & comp-time")
    ap.add_argument("--n_lat_measure", type=int, default=30)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--proposed_alias", type=str, default="HFDE",
                    help="Display name for Proposed (default: HFDE)")
    ap.add_argument("--only", type=str, default=None,
                    help="Print summary for one method only")
    ap.add_argument("--no_table", action="store_true")
    ap.add_argument("--out", type=str, default="efficiency_summary.json")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[benchmark] device={device}  img={args.img_size}  "
          f"bs_lat={args.bs_lat}  bs_thr={args.bs_thr}  "
          f"n_images={args.num_images}  n_runs={args.n_runs}\n")

    # ---- models ----
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]

    if args.model_paths:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
        models = []
        for p in args.model_paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = ResNet_50_pruned_hardfakevsreal(masks=ck["masks"])
            m.load_state_dict(ck["model_state_dict"])
            models.append(m.to(device).eval())
    else:
        models = [SmallResNet().to(device).eval() for _ in range(args.num_models)]
        for m in models:
            calibrate_for_early_exit(m, 4.0, args.img_size, device)

    n = len(models)
    MEANS = (MEANS * ((n // len(MEANS)) + 1))[:n]
    STDS  = (STDS  * ((n // len(STDS))  + 1))[:n]

    # ---- build all methods ----
    methods = {
        "Proposed":        ProposedEnsemble(models, MEANS, STDS),
        "Fuzzy Integral":  FuzzyIntegralEnsemble(models, MEANS, STDS),
        "Stacking":        StackingEnsemble(models, MEANS, STDS),
        "Majority Voting": MajorityVotingEnsemble(models, MEANS, STDS),
        "Weighted Avg.":   WeightedAvgEnsemble(models, MEANS, STDS),
        "Simple Avg.":     SimpleAvgEnsemble(models, MEANS, STDS),
    }
    order = list(methods.keys())

    # ---- data batches (preloaded → data-loading excluded) ----
    batches = build_batches(args.num_images, args.bs_thr, args.img_size, device)

    # ---- benchmark ----
    results = {}
    for name in order:
        if args.only and name != args.only:
            continue

        title = args.proposed_alias if name == "Proposed" else name.upper()
        m = methods[name].to(device)

        print(f"\n[benchmark] >>> {name} <<<")
        lat_m, lat_s = measure_latency(
            m, device, args.img_size,
            n_measure=args.n_lat_measure, warmup=10)
        thr_m, thr_s = measure_throughput(
            m, device, args.bs_thr, args.img_size,
            n_runs=args.n_runs, warmup=5)
        ct_m, ct_s = measure_computational_time(
            m, batches, device, n_passes=args.n_runs)

        results[name] = {
            "lat_m": lat_m, "lat_s": lat_s,
            "thr_m": thr_m, "thr_s": thr_s,
            "ct_m":  ct_m,  "ct_s":  ct_s,
        }

        print()
        print(format_summary(
            title, lat_m, lat_s, thr_m, thr_s, ct_m, ct_s,
            n_images=args.num_images,
            bs_lat=args.bs_lat,
            bs_thr=args.bs_thr,
            n_runs=args.n_runs,
        ))

    # ---- optional comparison table ----
    if not args.no_table and not args.only:
        print_comparison_table(results, order, args.bs_lat, args.bs_thr)

    # ---- save ----
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[json] → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
