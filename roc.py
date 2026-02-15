import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import argparse
import os
import json

# ایمپورت‌های پروژه
from dataset_utils import create_dataloaders
from main_code import FuzzyHesitantEnsemble, load_pruned_models

def get_ensemble_preds(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference"):
            images = images.to(device)
            
            # خروجی مدل انسمبل: (output, weights)
            output, _ = model(images)
            
            probs = torch.sigmoid(output.squeeze())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='./roc_results')
    
    # لیست فایل‌های انسمبل (فرمت JSON)
    # مثال: '[{"path":"run1/final.pt", "name":"Fuzzy Run 1"}, ...]'
    parser.add_argument('--models_config', type=str, required=True)
    
    # مسیر مدل‌های پایه (ResNet های پراپ شده)
    # نکته: فرض بر این است که مدل‌های پایه برای همه انسمبل‌ها یکسان است
    parser.add_argument('--base_models_paths', type=str, nargs='+', required=True)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. لود دیتاست
    print("Loading Test Loader...")
    _, _, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=False, seed=42, is_main=True
    )
    
    # 2. لود مدل‌های پایه (فقط یک بار)
    print("Loading Base Models...")
    base_models = load_pruned_models(args.base_models_paths, device, is_main=True)
    num_base_models = len(base_models)
    
    # پیکربندی مدل‌ها برای رسم
    models_list = json.loads(args.models_config)
    
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'green', 'blue', 'red']
    
    for i, config in enumerate(models_list):
        model_path = config['path']
        model_name = config['name']
        
        print(f"\nProcessing: {model_name} from {model_path}")
        
        if not os.path.exists(model_path):
            print(f"Warning: File not found {model_path}")
            continue
            
        # 3. لود چک‌پوینت
        ckpt = torch.load(model_path, map_location=device)
        
        # استخراج means و stds از داخل فایل (اگر موجود نبود، پیش‌فرض می‌گیرد)
        if 'means' in ckpt and 'stds' in ckpt:
            means = ckpt['means']
            stds = ckpt['stds']
        else:
            print("  Warning: means/stds not found in file, using defaults.")
            means = [(0.5207, 0.4258, 0.3806)] * num_base_models
            stds = [(0.2490, 0.2239, 0.2212)] * num_base_models

        # 4. ساخت معماری انسمبل
        ensemble = FuzzyHesitantEnsemble(
            models=base_models, 
            means=means, 
            stds=stds
        ).to(device)
        
        # 5. لود وزن‌های آموزش‌دیده (State Dict)
        # این بخش با فرمت ذخیره‌سازی شما هماهنگ است
        if 'ensemble_state_dict' in ckpt:
            ensemble.load_state_dict(ckpt['ensemble_state_dict'])
        else:
            ensemble.load_state_dict(ckpt)
            
        # 6. استخراج پیش‌بینی‌ها
        y_true, y_scores = get_ensemble_preds(ensemble, test_loader, device)
        
        # 7. محاسبه و رسم ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # آزادسازی حافظه
        del ensemble
        torch.cuda.empty_cache()

    # تنظیمات نمودار
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of Ensemble Methods')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'ensemble_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
