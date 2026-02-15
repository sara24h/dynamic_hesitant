import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))
    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class HesitantFuzzyMembership(nn.Module):
    def __init__(self, input_dim=128, num_models=3, num_memberships=3, dropout=0.3):
        super().__init__()
        self.num_models = num_models
        self.num_memberships = num_memberships
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.membership_generator = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_models * num_memberships)
        )
        self.aggregation_weights = nn.Parameter(torch.ones(num_memberships) / num_memberships)

    def forward(self, x):
        features = self.feature_net(x).flatten(1)
        memberships = self.membership_generator(features)
        memberships = memberships.view(-1, self.num_models, self.num_memberships)
        memberships = torch.sigmoid(memberships)
        agg_weights = F.softmax(self.aggregation_weights, dim=0)
        final_weights = (memberships * agg_weights.view(1, 1, -1)).sum(dim=2)
        final_weights = F.softmax(final_weights, dim=1)
        return final_weights, memberships

class FuzzyHesitantEnsemble(nn.Module):
    def __init__(self, models, means, stds, num_memberships=3, freeze_models=True, 
                 cum_weight_threshold=0.9, hesitancy_threshold=0.2):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128, num_models=self.num_models, num_memberships=num_memberships)
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters(): p.requires_grad = False

    def _compute_mask_vectorized(self, final_weights, avg_hesitancy):
        batch_size = final_weights.size(0)
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
        mask = (cum_weights <= self.cum_weight_threshold).float()
        mask[:, 0] = 1.0
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
        mask = torch.where(high_hesitancy_mask, torch.ones_like(mask), mask)
        final_mask = torch.zeros_like(final_weights)
        final_mask.scatter_(1, sorted_indices, mask)
        return final_mask

    def forward(self, x, return_details=False):
        final_weights, all_memberships = self.hesitant_fuzzy(x)
        hesitancy = all_memberships.var(dim=2)
        avg_hesitancy = hesitancy.mean(dim=1)
        mask = self._compute_mask_vectorized(final_weights, avg_hesitancy)
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)

        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        active_models = torch.any(final_weights > 0, dim=0).nonzero(as_tuple=True)[0]

        for i in active_models:
            x_n = self.normalizations(x, i.item())
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)): out = out[0]
            outputs[:, i] = out

        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
        if return_details: return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


def load_resnet_model(model_class, state_dict, device):
    """ساخت مدل و لود وزن‌ها"""
    model = model_class(masks=state_dict.get('masks', None))
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device).eval()
    return model

def load_ensemble_checkpoint(ckpt_path, base_model_paths, device):
  
    print(f"  -> Loading checkpoint: {os.path.basename(ckpt_path)}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    means = checkpoint['means']
    stds = checkpoint['stds']
    
    # لود مدل‌های پایه
    base_models = []
    try:
        # سعی می‌کنیم کلاس مدل پایه را ایمپورت کنیم
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        print("ERROR: Could not import 'ResNet_50_pruned_hardfakevsreal'. Make sure model definition exists.")
        raise

    for path in base_model_paths:
        if not os.path.exists(path):
            print(f"ERROR: Base model file not found at {path}")
            raise FileNotFoundError(f"Base model missing: {path}")
        
        base_ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = load_resnet_model(ResNet_50_pruned_hardfakevsreal, base_ckpt, device)
        base_models.append(model)
        
    # ساخت انسمبل
    ensemble = FuzzyHesitantEnsemble(
        base_models, means, stds, 
        num_memberships=3, 
        freeze_models=True
    ).to(device)
    
    # لود وزن‌های لایه تصمیم‌گیرنده
    ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
    ensemble.eval()
    
    return ensemble

def main():
    parser = argparse.ArgumentParser(description="Plot ROC Curves for Multiple Models")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to test dataset root (with class folders)")
    parser.add_argument('--base_models', type=str, nargs='+', required=True, help="Paths to base pruned models (.pt)")
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help="Paths to ensemble checkpoint files (.pt)")
    parser.add_argument('--names', type=str, nargs='+', required=True, help="Legend names for each checkpoint")
    parser.add_argument('--output', type=str, default='roc_comparison.png', help="Output plot filename")
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(args.checkpoints) != len(args.names):
        print("Error: Number of --checkpoints must match number of --names")
        return

    # 1. آماده‌سازی دیتاست تست
    print("Loading Test Dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # نکته: نرمالایزیشن اینجا مهم نیست چون مدل خودش داخلش نرمال می‌کند
    ])
    
    # فرض بر این است که دیتاست با ImageFolder لود می‌شود
    # اگر دیتاست شما ساختار خاصی دارد، اینجا را تغییر دهید
    if os.path.exists(os.path.join(args.data_dir, 'test')):
        data_path = os.path.join(args.data_dir, 'test')
    else:
        data_path = args.data_dir
        
    dataset = ImageFolder(root=data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. تنظیمات نمودار
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'cyan', 'magenta', 'yellow']
    
    # 3. پردازش هر مدل
    for i, ckpt_path in enumerate(args.checkpoints):
        model_name = args.names[i]
        print(f"\nProcessing Model {i+1}/{len(args.checkpoints)}: {model_name}")
        
        try:
            # لود مدل
            model = load_ensemble_checkpoint(ckpt_path, args.base_models, device)
            
            # اینفرنس
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=f"Inferring {model_name}"):
                    images = images.to(device)
                    logits, _ = model(images)
                    probs = torch.sigmoid(logits.squeeze(1))
                    
                    all_labels.append(labels)
                    all_probs.append(probs.cpu())
            
            y_true = np.concatenate(all_labels)
            y_scores = np.concatenate(all_probs)
            
            # محاسبه ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # رسم
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
                     label=f'{model_name} (AUC = {roc_auc:.4f})')
            
            # پاک کردن حافظه
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # 4. نهایی کردن نمودار
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nROC Curve saved to: {args.output}")

if __name__ == "__main__":
    main()
