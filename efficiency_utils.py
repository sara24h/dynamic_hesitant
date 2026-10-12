import time
import numpy as np
import torch

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class Stopwatch:
    """Wall-clock timer that synchronizes the GPU on start/stop."""

    def __init__(self, device):
        self.device = device
        self.t0 = None

    def start(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def stop(self):
        """Seconds since start(). Can be called several times."""
        _sync(self.device)
        return time.perf_counter() - self.t0


def environment_info(device):
    info = {'torch_version': torch.__version__, 'device': str(device)}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info.update({
            'gpu_name': props.name,
            'gpu_total_mem_gb': round(props.total_memory / 2 ** 30, 2),
            'visible_gpu_count': torch.cuda.device_count(),
            'gpus_used_by_script': 1,
            'cudnn_version': torch.backends.cudnn.version(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        })
    return info


# ---------------------------------------------------- training-time summary
def summarize_training_times(epoch_times, train_times, val_times, total_time):
    et = np.asarray(epoch_times, dtype=float)
    return {
        'total_training_time_s': float(total_time),
        'total_training_time_min': float(total_time / 60.0),
        'num_epochs': int(len(et)),
        'mean_epoch_time_s': float(et.mean()),
        # epoch 1 contains CUDA/cuDNN initialisation, so report both
        'mean_epoch_time_excl_first_s': float(et[1:].mean()) if len(et) > 1 else float(et.mean()),
        'mean_train_phase_s': float(np.mean(train_times)),
        'mean_val_phase_s': float(np.mean(val_times)),
        'epoch_times_s': [float(v) for v in et],
    }


# ------------------------------------------------------------ measurements
@torch.no_grad()
def measure_latency(fn, samples, device, warmup=50):
    """
    Per-sample latency (batch size = 1).
    samples: list of tensors (1,C,H,W) already on `device`.
    Returns (stats_dict, list_of_outputs).
    """
    for k in range(warmup):
        fn(samples[k % len(samples)])
    _sync(device)

    times_ms, outs = [], []
    use_cuda = device.type == 'cuda'
    for x in samples:
        if use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(x)
            e.record()
            torch.cuda.synchronize(device)
            times_ms.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            out = fn(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        outs.append(out)

    t = np.asarray(times_ms, dtype=float)
    stats = {
        'num_samples': int(len(t)),
        'mean_ms': float(t.mean()),
        'std_ms': float(t.std(ddof=1)) if len(t) > 1 else 0.0,
        'median_ms': float(np.median(t)),
        'p95_ms': float(np.percentile(t, 95)),
        'p99_ms': float(np.percentile(t, 99)),
        'min_ms': float(t.min()),
        'max_ms': float(t.max()),
        'throughput_bs1_img_per_s': float(1000.0 / t.mean()),
    }
    return stats, outs


@torch.no_grad()
def measure_throughput(fn, batches, device, warmup_iters=10, num_iters=100):
    """
    Throughput on pre-loaded GPU batches (cycled if num_iters > len(batches)).
    images/s = total images / total (synchronized) wall time.
    """
    for k in range(warmup_iters):
        fn(batches[k % len(batches)])
    _sync(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        base_mem = torch.cuda.memory_allocated(device)

    n_img = 0
    _sync(device)
    t0 = time.perf_counter()
    for k in range(num_iters):
        x = batches[k % len(batches)]
        fn(x)
        n_img += x.size(0)
    _sync(device)
    elapsed = time.perf_counter() - t0

    res = {
        'batch_size': int(batches[0].size(0)),
        'num_iters': int(num_iters),
        'num_images': int(n_img),
        'total_time_s': float(elapsed),
        'throughput_img_per_s': float(n_img / elapsed),
        'ms_per_batch': float(1000.0 * elapsed / num_iters),
        'ms_per_image': float(1000.0 * elapsed / n_img),
    }
    if device.type == 'cuda':
        # activation / workspace memory only (excludes weights and cached inputs)
        res['extra_peak_mem_mb'] = float(
            (torch.cuda.max_memory_allocated(device) - base_mem) / 2 ** 20)
    return res


@torch.no_grad()
def measure_full_pass(fn, loader, device):
    """
    One pass over the WHOLE test loader.
      model_compute_time_s : time spent in the model only (data loading excluded)
      total_time_s         : wall time of the pass (INCLUDES data loading)
    """
    compute_s, n = 0.0, 0
    _sync(device)
    t_wall = time.perf_counter()
    for images, _ in loader:
        images = images.to(device)
        _sync(device)
        t0 = time.perf_counter()
        fn(images)
        _sync(device)
        compute_s += time.perf_counter() - t0
        n += images.size(0)
    wall = time.perf_counter() - t_wall
    return {'num_images': int(n),
            'model_compute_time_s': float(compute_s),
            'img_per_s_compute': float(n / compute_s),
            'total_time_s': float(wall),
            'img_per_s_end_to_end': float(n / wall)}


# ------------------------------------------------------------ main benchmark
def benchmark_efficiency(ensemble, test_loader, device, model_names,
                         latency_samples=300, latency_warmup=50,
                         throughput_batches=20, throughput_warmup=10,
                         throughput_iters=100, verbose=False):
  
    ensemble.eval()
    dataset = test_loader.dataset
    n_models = len(ensemble.models)
    norm = ensemble.normalizations

    # ---- cache real test data on GPU --------------------------------------
    n_lat = min(latency_samples, len(dataset))
    lat_inputs = [dataset[i][0].unsqueeze(0).to(device) for i in range(n_lat)]

    thr_inputs = []
    for images, _ in test_loader:
        thr_inputs.append(images.to(device))
        if len(thr_inputs) >= throughput_batches:
            break
    batch_size = thr_inputs[0].size(0)

    # ---- callables to benchmark -------------------------------------------
    methods = {
        'HFDE (dynamic)': lambda x: ensemble(x),
        'Router only': lambda x: ensemble.hesitant_fuzzy(x),
    }
    for i in range(n_models):
        name = model_names[i] if i < len(model_names) else f'model_{i + 1}'
        methods[f'Single: {name}'] = (lambda x, i=i: ensemble.models[i](norm(x, i)))
    methods['Static ensemble (all models)'] = lambda x: torch.stack(
        [_first(ensemble.models[i](norm(x, i))) for i in range(n_models)]).mean(0)

    results = {
        'environment': environment_info(device),
        'protocol': {
            'latency_batch_size': 1,
            'latency_warmup_iters': latency_warmup,
            'latency_num_samples': n_lat,
            'throughput_batch_size': int(batch_size),
            'throughput_warmup_iters': throughput_warmup,
            'throughput_iters': throughput_iters,
            'timing': 'torch.cuda.Event + synchronize, inputs pre-loaded on GPU, no_grad',
        },
        'methods': {},
    }

    print("\n" + "=" * 70)
    print("EFFICIENCY BENCHMARK (Latency / Throughput)")
    print("=" * 70)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)} (single GPU)")
    if verbose:
        print(f"Latency: batch=1, {n_lat} samples, warm-up={latency_warmup} | "
              f"Throughput: batch={batch_size}, {throughput_iters} iters, warm-up={throughput_warmup}")

    hfde_outs = None
    for name, fn in methods.items():
        lat, outs = measure_latency(fn, lat_inputs, device, warmup=latency_warmup)
        thr = measure_throughput(fn, thr_inputs, device,
                                 warmup_iters=throughput_warmup, num_iters=throughput_iters)
        results['methods'][name] = {'latency': lat, 'throughput': thr}
        if name == 'HFDE (dynamic)':
            hfde_outs = outs
        if verbose:
            print(f"  {name:<34} lat {lat['mean_ms']:7.2f} ± {lat['std_ms']:5.2f} ms | "
                  f"{lat['throughput_bs1_img_per_s']:7.1f} img/s (bs=1) | "
                  f"{thr['throughput_img_per_s']:8.1f} img/s (bs={batch_size})")
        del outs

    # ---- dynamic-routing statistics (what is ACTUALLY executed) -----------
    # forward() runs a model iff its final weight is exactly > 0 (binary mask).
    w = torch.cat([o[1] for o in hfde_outs], dim=0)            # (N, M)
    active = (w > 0)
    results['dynamic_routing'] = {
        'mean_active_models_per_sample_bs1': float(active.sum(1).float().mean().item()),
        'num_models': int(n_models),
        'activation_rate_pct_bs1': {
            (model_names[i] if i < len(model_names) else f'model_{i + 1}'):
                float(active[:, i].float().mean().item() * 100.0)
            for i in range(n_models)},
    }
    with torch.no_grad():
        executed = [int((ensemble(x)[1] > 0).any(dim=0).sum().item()) for x in thr_inputs]
    results['dynamic_routing'][f'mean_models_executed_per_batch_bs{batch_size}'] = float(np.mean(executed))
    if verbose:
        print(f"\n  HFDE avg. active models / sample (bs=1): "
              f"{results['dynamic_routing']['mean_active_models_per_sample_bs1']:.2f} of {n_models}")
        print(f"  HFDE avg. models executed / batch (bs={batch_size}): {np.mean(executed):.2f} of {n_models}")

    # ---- full test pass (end-to-end, includes data loading) ---------------
    results['full_test_pass_hfde'] = measure_full_pass(methods['HFDE (dynamic)'], test_loader, device)
    fp = results['full_test_pass_hfde']
    if verbose:
        print(f"  Full test pass (HFDE, incl. data loading): {fp['num_images']} images in "
              f"{fp['total_time_s']:.2f}s ({fp['img_per_s_end_to_end']:.1f} img/s)")
    print("  (details of all methods -> efficiency_report.json)")
    print("=" * 70 + "\n")

    del lat_inputs, thr_inputs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


def print_headline(results):
    """Print ONLY the three headline numbers for HFDE."""
    h = results['methods']['HFDE (dynamic)']
    bs = h['throughput']['batch_size']
    print("=" * 70)
    print("HFDE EFFICIENCY SUMMARY")
    print("=" * 70)
    print(f"Latency (bs=1)          : {h['latency']['mean_ms']:.2f} ± {h['latency']['std_ms']:.2f} ms / image")
    print(f"Throughput (bs={bs})".ljust(24) + f": {h['throughput']['throughput_img_per_s']:.1f} images/s")
    fp = results['full_test_pass_hfde']
    print(f"Computational time      : {fp['model_compute_time_s']:.2f} s "
          f"(inference on the whole test set, {fp['num_images']} images)")
    print("=" * 70 + "\n")
