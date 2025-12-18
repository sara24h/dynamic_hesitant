import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
import argparse
import json
import shutil
import warnings
from tqdm import tqdm
import numpy as np

# Import custom modules
from utils import set_seed
from datasets.base_dataset import create_dataloaders_ddp, set_seed
from models import FuzzyHesitantEnsemble, load_pruned_models

warnings.filterwarnings("ignore")


def setup_ddp():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate_single_model(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    name: str, 
    rank: int
) -> float:
    """Evaluate a single model's accuracy"""
    model.eval()
    correct = total = 0
 
    iterator = tqdm(loader, desc=f"Evaluating {name}") if rank == 0 else loader
 
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device).float()
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        pred = (out.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
 
    acc = 100. * correct / total
 
    if rank == 0:
        print(f" {name}: {acc:.2f}%")
 
    return acc


def train_hesitant_fuzzy_ddp(
    ensemble_model, 
    train_loader, 
    val_loader, 
    num_epochs, 
    lr, 
    device, 
    save_dir, 
    rank, 
    world_size
):
    """Train the hesitant fuzzy membership network"""
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
 
    hesitant_net = ensemble_model.module.hesitant_fuzzy
    optimizer = torch.optim.AdamW(hesitant_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCEWithLogitsLoss()
   
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'membership_variance': []}
   
    if rank == 0:
        print("="*70)
        print("Training Fuzzy Hesitant Network (DDP)")
        print("="*70)
        print(f"Trainable params: {sum(p.numel() for p in hesitant_net.parameters()):,}")
        print(f"World Size (GPUs): {world_size}")
        print(f"Epochs: {num_epochs} | Initial LR: {lr}")
        print(f"Hesitant memberships per model: {hesitant_net.num_memberships}\n")
   
    for epoch in range(num_epochs):
        ensemble_model.train()
        train_loader.sampler.set_epoch(epoch)
     
        train_loss = train_correct = train_total = 0.0
        membership_vars = []
     
        iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]') if rank == 0 else train_loader
       
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device).float()
         
            optimizer.zero_grad()
            outputs, weights, memberships, _ = ensemble_model(images, return_details=True)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()
           
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            pred = (outputs.squeeze(1) > 0).long()
            train_correct += pred.eq(labels.long()).sum().item()
            train_total += batch_size
         
            membership_vars.append(memberships.var(dim=2).mean().item())
           
            if rank == 0:
                current_acc = 100. * train_correct / train_total
                avg_loss = train_loss / train_total
                iterator.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
       
        # Synchronize metrics across all GPUs
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_correct_tensor = torch.tensor(train_correct).to(device)
        train_total_tensor = torch.tensor(train_total).to(device)
     
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)
     
        train_acc = 100. * train_correct_tensor.item() / train_total_tensor.item()
        train_loss = train_loss_tensor.item() / train_total_tensor.item()
        avg_membership_var = np.mean(membership_vars)
       
        val_acc = evaluate_accuracy_ddp(ensemble_model, val_loader, device, rank)
        scheduler.step()
     
        if rank == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['membership_variance'].append(avg_membership_var)
           
            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f" Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f" Membership Variance (Hesitancy): {avg_membership_var:.4f}")
           
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                tmp_path = os.path.join(save_dir, 'best_tmp.pt')
                final_path = os.path.join(save_dir, 'best_hesitant_fuzzy.pt')
               
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'hesitant_state_dict': hesitant_net.state_dict(),
                        'val_acc': val_acc,
                        'history': history
                    }, tmp_path)
                   
                    if os.path.exists(tmp_path):
                        shutil.move(tmp_path, final_path)
                        print(f" Best model saved → {val_acc:.2f}%")
                except Exception as e:
                    print(f" [ERROR] Failed to save model: {e}")
           
            print("-" * 70)
       
        dist.barrier()
   
    if rank == 0:
        print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
 
    return best_val_acc, history


@torch.no_grad()
def evaluate_ensemble_final_ddp(model, loader, device, name, model_names, rank):
    """Evaluate ensemble with detailed statistics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_weights = []
    all_memberships = []
    
    activation_counts = torch.zeros(len(model_names), device=device)
    total_samples = 0

    iterator = tqdm(loader, desc=f"Evaluating {name}", leave=True) if rank == 0 else loader

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        outputs, weights, memberships, _ = model(images, return_details=True)
        pred = (outputs.squeeze(1) > 0).long()

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_weights.append(weights.cpu())
        all_memberships.append(memberships.cpu())

        active_per_model = (weights > 1e-4).sum(dim=0).float()
        activation_counts += active_per_model
        total_samples += images.size(0)

    # Synchronize across GPUs
    dist.all_reduce(activation_counts, op=dist.ReduceOp.SUM)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    total_samples = total_samples_tensor.item()

    all_weights = torch.cat(all_weights).cpu().numpy()
    all_memberships = torch.cat(all_memberships).cpu().numpy()
    activation_counts = activation_counts.cpu().numpy()

    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    avg_weights = all_weights.mean(axis=0)
    activation_percentages = (activation_counts / total_samples) * 100

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{name.upper()} SET RESULTS")
        print(f"{'='*70}")
        print(f" → Accuracy: {acc:.3f}%")
        print(f" → Total Samples: {total_samples:,}")

        print(f"\nAverage Model Weights:")
        for i, (w, name) in enumerate(zip(avg_weights, model_names)):
            print(f"   {i+1:2d}. {name:<25}: {w:6.4f} ({w*100:5.2f}%)")

        print(f"\nActivation Frequency:")
        for i, (perc, count, name) in enumerate(zip(activation_percentages, activation_counts, model_names)):
            print(f"   {i+1:2d}. {name:<25}: {perc:6.2f}% active ({int(count):,} / {total_samples:,} samples)")

        print(f"\nHesitant Membership Values:")
        for i, name in enumerate(model_names):
            mems = all_memberships[:, i].mean(axis=0)
            var = all_memberships[:, i].var(axis=0).mean()
            print(f"   {i+1:2d}. {name:<25}: μ = [{', '.join([f'{m:.3f}' for m in mems])}] | Hesitancy = {var:.4f}")

        print(f"{'='*70}")

    return acc, avg_weights.tolist(), all_memberships.mean(axis=0).tolist(), activation_percentages.tolist()


@torch.no_grad()
def evaluate_accuracy_ddp(model, loader, device, rank):
    """Quick accuracy evaluation"""
    model.eval()
    correct = total = 0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs, _ = model(images)
        pred = (outputs.squeeze(1) > 0).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()
 
    acc = 100. * correct / total
    return acc


def main():
    SEED = 42
    set_seed(SEED)
   
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)
   
    parser = argparse.ArgumentParser(description="Train Fuzzy Hesitant Ensemble with DDP")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_memberships', type=int, default=3)
   
    parser.add_argument('--dataset', type=str, 
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'], 
                       required=True)
    parser.add_argument('--data_dir', type=str, required=True)
   
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
   
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/checkpoints')
    parser.add_argument('--seed', type=int, default=SEED)
   
    args = parser.parse_args()
   
    if len(args.model_names) != len(args.model_paths):
        raise ValueError(
            f"Number of model_names ({len(args.model_names)}) "
            f"must match model_paths ({len(args.model_paths)})"
        )
   
    # Default normalization parameters (adjust as needed)
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
   
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
   
    if args.seed != SEED:
        set_seed(args.seed)
   
    if is_main:
        print(f"="*70)
        print(f"Multi-GPU Training with DDP | SEED: {args.seed}")
        print(f"="*70)
        print(f"World Size: {world_size} GPUs")
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print(f"Models to load: {len(args.model_paths)}")
        print(f"="*70 + "\n")
   
    # Load models
    base_models = load_pruned_models(args.model_paths, device, rank)
 
    if len(base_models) != len(args.model_paths):
        if is_main:
            print(f"[WARNING] Only {len(base_models)}/{len(args.model_paths)} models loaded.")
        MEANS = MEANS[:len(base_models)]
        STDS = STDS[:len(base_models)]
        MODEL_NAMES = args.model_names[:len(base_models)]
    else:
        MODEL_NAMES = args.model_names
   
    # Create ensemble
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True,
        cum_weight_threshold=0.9,
        hesitancy_threshold=0.2
    ).to(device)
 
    ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=False)
 
    if is_main:
        hesitant_net = ensemble.module.hesitant_fuzzy
        trainable = sum(p.numel() for p in hesitant_net.parameters())
        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable:,}\n")
   
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders_ddp(
        args.data_dir, args.batch_size, rank, world_size, dataset_type=args.dataset
    )
 
    # Evaluate individual models
    if is_main:
        print("\n" + "="*70)
        print("EVALUATING INDIVIDUAL MODELS (Before Training)")
        print("="*70)
        individual_accs = []
        for i, model in enumerate(base_models):
            acc = evaluate_single_model(model, test_loader, device, 
                                       f"Model {i+1} ({MODEL_NAMES[i]})", rank)
            individual_accs.append(acc)
        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single Model: {MODEL_NAMES[best_idx]} → {best_single:.2f}%")
     
    dist.barrier()
   
    # Train ensemble
    best_val_acc, history = train_hesitant_fuzzy_ddp(
        ensemble, train_loader, val_loader,
        args.epochs, args.lr, device, args.save_dir, rank, world_size
    )
 
    # Load best model
    ckpt_path = os.path.join(args.save_dir, 'best_hesitant_fuzzy.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ensemble.module.hesitant_fuzzy.load_state_dict(ckpt['hesitant_state_dict'])
        if is_main:
            print("Best model loaded.\n")
 
    dist.barrier()
 
    # Final evaluation
    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)
        ensemble_test_acc, ensemble_weights, membership_values, activation_percentages = \
            evaluate_ensemble_final_ddp(ensemble, test_loader, device, "Test", MODEL_NAMES, rank)
       
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Hesitant Ensemble: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
       
        # Save results
        results = {
            "method": "Fuzzy Hesitant Sets (DDP)",
            "dataset": args.dataset,
            "seed": args.seed,
            "num_gpus": world_size,
            "num_memberships": args.num_memberships,
            "model_names": MODEL_NAMES,
            "individual_accuracies": {MODEL_NAMES[i]: acc for i, acc in enumerate(individual_accs)},
            "best_single": {"name": MODEL_NAMES[best_idx], "acc": best_single},
            "ensemble": {
                "acc": ensemble_test_acc,
                "weights": ensemble_weights,
                "activation_percentages": activation_percentages
            },
            "improvement": ensemble_test_acc - best_single,
            "training_history": history
        }
       
        result_path = os.path.join(args.save_dir, 'results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {result_path}")
        print("="*70)
   
    cleanup_ddp()


if __name__ == "__main__":
    main()
