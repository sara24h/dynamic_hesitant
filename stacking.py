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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from metrics_utils import plot_roc_and_f1

warnings.filterwarnings("ignore")
from dataset_utils import (
    UADFVDataset,
    CustomGenAIDataset,
    create_dataloaders,
    get_sample_info,
    worker_init_fn
)
from visualization_utils import GradCAM, generate_lime_explanation, generate_visualizations

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

# ================== STACKING META LEARNER (Logistic Regression) ==================
class StackingMetaLearner(nn.Module):
    def __init__(self, num_models: int):
        super().__init__()
        self.linear = nn.Linear(num_models, 1)
    def forward(self, x):
        return self.linear(x)

# ================== STACKING ENSEMBLE CLASS ==================
class StackingEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]],
                 stds: List[Tuple[float]], freeze_models: bool = True):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.meta_learner = StackingMetaLearner(self.num_models)
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor, return_details: bool = False):
        logits_list = []
        
        # اجرای مدل‌ها (نکته: برای DDP مدل‌ها روی همان دستگاه فرآیند هستند)
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            # چون مدل‌ها فریز شده‌اند، no_grad برای صرفه‌جویی حافظه بهتر است
            # اما در DDP برای backward نهایی (اگر weights trainable باشند) نباید no_grad کلی گرفت
            # اینجا فقط مدل‌های پایه داخل no_grad هستند
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            logits_list.append(out)
       
        stacked_logits = torch.cat(logits_list, dim=1)
        final_output = self.meta_learner(stacked_logits)

        if return_details:
            batch_size = x.size(0)
            dummy_weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            dummy_memberships = torch.zeros(batch_size, self.num_models, 3, device=x.device)
            return final_output, dummy_weights, dummy_memberships, stacked_logits
        else:
            batch_size = x.size(0)
            dummy_weights = torch.ones(batch_size, self.num_models, device=x.device) / self.num_models
            return final_output, dummy_weights

# ================== MODEL LOADING ==================
def load_pruned_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")
    models = []
    if is_main:
        print(f"Loading {len(model_paths)} pruned models...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main:
                print(f" [WARNING] File not found: {path}")
            continue
        if is_main:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
            param_count = sum(p.numel() for p in model.parameters())
            if is_main:
                print(f" → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
    if len(models) == 0:
        raise ValueError("No models loaded!")
    if is_main:
        print(f"All {len(models)} models loaded!\n")
    return models

# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device: torch.device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item()
    if is_main:
        print(f" {name}: {acc:.2f}%")
    return acc

@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item()
    return acc

@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, is_main=True):
    model.eval()
    total_correct = 0
    total_samples = 0
    if is_main:
        print(f"\nEvaluating {name} set (Stacking - Logistic Regression)...")
   
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images, return_details=False)
        pred = (outputs.squeeze(1) > 0).long()
        total_correct += pred.eq(labels.long()).sum().item()
        total_samples += labels.size(0)
    stats = torch.tensor([total_correct, total_samples], dtype=torch.long, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if is_main:
        total_correct = stats[0].item()
        total_samples = stats[1].item()
        acc = 100. * total_correct / total_samples
       
        meta_learner = model.meta_learner
        weights = meta_learner.linear.weight.data.cpu().squeeze().numpy()
        bias = meta_learner.linear.bias.data.cpu().item()
        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS (Stacking - Logistic Regression)")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")
       
        print(f"\nMeta-Learner (Logistic Regression) Analysis:")
        print(f" Bias (Intercept): {bias:+.4f}")
        print(f" Learned Weights (Importance of each base model):")
        for i, (w, mname) in enumerate(zip(weights, model_names)):
            print(f" {i+1:2d}. {mname:<25}: {w:+.4f}")
       
        print(f"{'='*70}")
        return acc, weights.tolist(), bias
    return 0.0, [0.0]*len(model_names), 0.0

def train_stacking(ensemble_model, train_loader, val_loader, num_epochs, lr,
                   device, save_dir, is_main, model_names):
    os.makedirs(save_dir, exist_ok=True)
    meta_learner = ensemble_model.module.meta_learner if hasattr(ensemble_model, 'module') else ensemble_model.meta_learner
   
    optimizer = torch.optim.AdamW(meta_learner.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    if is_main:
        print("="*70)
        print("Training Stacking Meta-Learner (Logistic Regression)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in meta_learner.parameters()):,}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Number of base models: {len(model_names)}\n")
    for epoch in range(num_epochs):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
       
        ensemble_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
       
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=not is_main):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
           
            outputs, _ = ensemble_model(images)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / train_total
       
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        if is_main:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, 'best_stacking_lr.pt')
            torch.save({
                'epoch': epoch + 1,
                'meta_state_dict': meta_learner.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, save_path)
            print(f"\n✓ Best model saved → {val_acc:.2f}%")
        if is_main:
            print()
    if is_main:
        print(f"\n{'='*70}")
        print(f"Training Completed!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*70}\n")
   
    return best_val_acc, history

# ================== DISTRIBUTED SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f"Distributed: rank {rank}/{world_size}, local_rank {local_rank}")
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ================== HELPER TO GET TEST INDICES ==================
def get_test_indices(test_loader):
    if hasattr(test_loader, 'sampler') and hasattr(test_loader.sampler, 'indices'):
        return test_loader.sampler.indices
    elif hasattr(test_loader.dataset, 'indices'):
        return test_loader.dataset.indices
    else:
        return list(range(len(test_loader.dataset)))

# ================== FINAL EVALUATION & ROC ==================
@torch.no_grad()
def final_evaluation_and_roc(model, loader, device, save_dir, model_name, args, is_main):
    if not is_main or loader is None: return 0.0, None, None

    model.eval()
    
    base_dataset = loader.dataset
    if hasattr(base_dataset, 'dataset'): base_dataset = base_dataset.dataset
    
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'):
        test_indices = loader.sampler.indices
    elif hasattr(loader.dataset, 'indices'):
        test_indices = loader.dataset.indices
    else:
        test_indices = list(range(len(base_dataset)))

    all_y_true = []
    all_y_score = []
    all_y_pred = []
    lines = []

    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS")
    lines.append("="*100)
    
    TP, TN, FP, FN = 0, 0, 0, 0
    correct_count = 0

    print(f"\nRunning Final Evaluation on {len(test_indices)} samples...")
    for i, global_idx in enumerate(tqdm(test_indices, desc="Final Eval")):
        try:
            image, label = base_dataset[global_idx]
            path, _ = get_sample_info(base_dataset, global_idx)
        except: continue

        image = image.unsqueeze(0).to(device)
        label_int = int(label)
        
        output = model(image)
        if isinstance(output, (tuple, list)): output = output[0]
        prob = torch.sigmoid(output.squeeze()).item()
        pred_int = int(prob > 0.5)
        
        all_y_true.append(label_int)
        all_y_score.append(prob)
        all_y_pred.append(pred_int)
        
        if pred_int == label_int: correct_count += 1
        if label_int == 1:
            if pred_int == 1: TP += 1
            else: FN += 1
        else:
            if pred_int == 1: FP += 1
            else: TN += 1
            
        filename = os.path.basename(path)
        line = f"{i+1:<10} {filename:<60} {label_int:<12} {pred_int:<15} {'Yes' if pred_int==label_int else 'No':<10}"
        lines.append(line)

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Print
    print(f"\nPrecision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {TP:<15} {FN:<15}")
    print(f"    Actual Fake      {FP:<15} {TN:<15}")
    print(f"\nCorrect Predictions: {correct_count} ({acc*100:.2f}%)")
    print(f"Incorrect Predictions: {total - correct_count} ({(1-acc)*100:.2f}%)")

    # Save Log
    output_str = []
    output_str.append(f"Accuracy: {acc*100:.2f}%")
    output_str.extend(lines)
    with open(os.path.join(save_dir, 'prediction_log.txt'), 'w') as f:
        f.write("\n".join(output_str))

    # Save ROC
    y_true_np = np.array(all_y_true)
    y_score_np = np.array(all_y_score)
    
    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    with open(roc_json_path, 'w') as f:
        json.dump({
            "metadata": {"seed": args.seed, "dataset": args.dataset},
            "y_true": y_true_np.tolist(),
            "y_score": y_score_np.tolist()
        }, f, indent=2)
    
    roc_txt_path = os.path.join(save_dir, "roc_data_test.txt")
    with open(roc_txt_path, 'w') as f:
        f.write("y_true\ty_score\n")
        for t, s in zip(y_true_np, y_score_np):
            f.write(f"{int(t)}\t{s:.6f}\n")
            
    return acc * 100, y_true_np, y_score_np

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Stacking Ensemble Training with Logistic Regression")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'deepfake_lab', 'hard_fake_real', 'custom_genai', 'custom_genai_v2', 'uadfV', 'real_fake_dataset'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, nargs='+', required=True, help='List of seeds to run')
   
    args = parser.parse_args()
    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    for current_seed in args.seed:
        print(f"\n{'#'*80}")
        print(f"# STARTING RUN FOR SEED: {current_seed}")
        print(f"{'#'*80}\n")
        set_seed(current_seed)
        device, local_rank, rank, world_size = setup_distributed()
        is_main = rank == 0
        current_save_dir = os.path.join(args.save_dir, f'seed_{current_seed}')
       
        if is_main:
            print("="*70)
            print(f"STACKING ENSEMBLE TRAINING (Logistic Regression)")
            print(f"Distributed on {world_size} GPU(s) | Seed: {current_seed}")
            print("="*70)
            print(f"Dataset: {args.dataset}")
            print(f"Data directory: {args.data_dir}")
            print(f"Batch size: {args.batch_size}")
            print(f"Models: {len(args.model_paths)}")
            print(f"Output Dir: {current_save_dir}")
            print("="*70 + "\n")

        MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
        STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
        MEANS = MEANS[:len(args.model_paths)]
        STDS = STDS[:len(args.model_paths)]

        base_models = load_pruned_models(args.model_paths, device, is_main)
        MODEL_NAMES = args.model_names[:len(base_models)]

        ensemble = StackingEnsemble(
            base_models, MEANS, STDS,
            freeze_models=True
        ).to(device)

        if world_size > 1:
            ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

        if is_main:
            trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
            total = sum(p.numel() for p in ensemble.parameters())
            print(f"Total params: {total:,} | Trainable: {trainable:,} | Frozen: {total-trainable:,}\n")

        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, args.batch_size, dataset_type=args.dataset,
            is_distributed=(world_size > 1), seed=current_seed, is_main=is_main)

        if is_main:
            print("\n" + "="*70)
            print("INDIVIDUAL MODEL PERFORMANCE (Before Training)")
            print("="*70)

        individual_accs = []
        for i, model in enumerate(base_models):
            acc = evaluate_single_model_ddp(
                model, test_loader, device,
                f"Model {i+1} ({MODEL_NAMES[i]})",
                MEANS[i], STDS[i], is_main)
            individual_accs.append(acc)

        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        if is_main:
            print(f"\nBest Single: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
            print("="*70)

        best_val_acc, history = train_stacking(
            ensemble, train_loader, val_loader,
            args.epochs, args.lr, device, current_save_dir, is_main, MODEL_NAMES)

        ckpt_path = os.path.join(current_save_dir, 'best_stacking_lr.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            meta_learner = ensemble.module.meta_learner if hasattr(ensemble, 'module') else ensemble.meta_learner
            meta_learner.load_state_dict(ckpt['meta_state_dict'])
            if is_main:
                print("Best model loaded.\n")

        ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble

        if is_main:
            os.makedirs(current_save_dir, exist_ok=True)
            # ساخت دیتالودر تست کامل فقط روی GPU اصلی برای ارزیابی نهایی
            _, _, test_loader_full = create_dataloaders(
                args.data_dir, args.batch_size, dataset_type=args.dataset,
                is_distributed=False, seed=current_seed, is_main=True)

            print("\n" + "="*70)
            print("FINAL ENSEMBLE EVALUATION")
            print("="*70)

            ensemble_test_acc, y_true, y_score = final_evaluation_and_roc(
                ensemble_module, test_loader_full, device, current_save_dir, 
                "Stacking_LR", args, is_main
            )

            print("\n" + "="*70)
            print("FINAL COMPARISON")
            print("="*70)
            print(f"Best Single Model: {best_single:.2f}%")
            print(f"Stacking Accuracy (LR): {ensemble_test_acc:.2f}%")
            print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
            print("="*70)

            final_results = {
                'seed': current_seed,
                'method': 'Stacking_LR',
                'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
                'ensemble': {'test_accuracy': float(ensemble_test_acc)},
                'improvement': float(ensemble_test_acc - best_single)
            }
            results_path = os.path.join(current_save_dir, 'final_results_stacking_lr.json')
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=4)

        cleanup_distributed()
        
        if is_main:
            plot_roc_and_f1(
                ensemble_module,
                test_loader_full, 
                device, 
                current_save_dir, 
                MODEL_NAMES,
                is_main
            )

if __name__ == "__main__":
    main()
