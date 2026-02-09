import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import os
import json
import argparse
from typing import List, Tuple
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

from dataset_utils import create_dataloaders
from metrics_utils import plot_roc_and_f1

# ================== AdaBoost Implementation ==================

class AdaBoostClassifier:
    """
    AdaBoost classifier for pre-trained PyTorch models
    """
    def __init__(self, models: List[nn.Module], device: torch.device, 
                 means: List[Tuple], stds: List[Tuple], model_names: List[str]):
        self.models = models
        self.device = device
        self.means = means
        self.stds = stds
        self.model_names = model_names
        self.n_models = len(models)
        
        # AdaBoost parameters
        self.model_weights = np.zeros(self.n_models)  # α (alpha) for each model
        self.sample_weights = None
        
        # Freeze all models
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
    
    def normalize_input(self, x: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Normalize input for specific model"""
        mean = torch.tensor(self.means[model_idx]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.stds[model_idx]).view(1, 3, 1, 1).to(self.device)
        return (x - mean) / std
    
    @torch.no_grad()
    def get_predictions(self, model_idx: int, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a specific model"""
        model = self.models[model_idx]
        model.eval()
        
        all_preds = []
        all_labels = []
        
        for images, labels in loader:
            images = images.to(self.device)
            images_norm = self.normalize_input(images, model_idx)
            
            outputs = model(images_norm)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            preds = (outputs.squeeze(1) > 0).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    def fit(self, train_loader: DataLoader, n_iterations: int = None):
        """
        Train AdaBoost using pre-trained models
        
        Args:
            train_loader: Training data loader
            n_iterations: Number of boosting iterations (default: number of models)
        """
        if n_iterations is None:
            n_iterations = self.n_models
        
        # Get all training data and labels
        print("\n" + "="*70)
        print("Collecting training data for AdaBoost...")
        print("="*70)
        
        all_data = []
        all_labels = []
        for images, labels in train_loader:
            all_data.append(images)
            all_labels.append(labels)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        n_samples = len(all_labels)
        
        # Initialize sample weights uniformly
        self.sample_weights = np.ones(n_samples) / n_samples
        
        print(f"Total training samples: {n_samples:,}\n")
        
        # AdaBoost iterations
        for t in range(min(n_iterations, self.n_models)):
            print(f"\n{'='*70}")
            print(f"AdaBoost Iteration {t+1}/{min(n_iterations, self.n_models)}")
            print(f"{'='*70}")
            print(f"Model: {self.model_names[t]}")
            
            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=n_samples,
                replacement=True
            )
            
            # Create weighted dataloader
            from torch.utils.data import TensorDataset
            weighted_dataset = TensorDataset(all_data, torch.from_numpy(all_labels))
            weighted_loader = DataLoader(
                weighted_dataset,
                batch_size=train_loader.batch_size,
                sampler=sampler,
                num_workers=0
            )
            
            # Get predictions from current model
            print("Getting predictions...")
            predictions, true_labels = self.get_predictions(t, weighted_loader)
            
            # Calculate weighted error
            incorrect = (predictions != true_labels).astype(int)
            weighted_error = np.sum(self.sample_weights * incorrect) / np.sum(self.sample_weights)
            
            # Avoid division by zero
            weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)
            
            # Calculate model weight (alpha)
            alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
            self.model_weights[t] = alpha
            
            print(f"Weighted Error: {weighted_error:.4f}")
            print(f"Model Weight (α): {alpha:.4f}")
            
            # Update sample weights
            self.sample_weights *= np.exp(-alpha * true_labels * (2 * predictions - 1))
            self.sample_weights /= np.sum(self.sample_weights)  # Normalize
            
            # Print weight statistics
            print(f"Sample weights - Min: {self.sample_weights.min():.6f}, "
                  f"Max: {self.sample_weights.max():.6f}, "
                  f"Mean: {self.sample_weights.mean():.6f}")
        
        # Normalize model weights
        print(f"\n{'='*70}")
        print("AdaBoost Training Completed!")
        print(f"{'='*70}")
        print("\nFinal Model Weights (α):")
        for i, (name, weight) in enumerate(zip(self.model_names[:n_iterations], 
                                                self.model_weights[:n_iterations])):
            print(f"  {i+1}. {name:<30}: {weight:.4f}")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using AdaBoost ensemble
        
        Returns:
            predictions: Final predictions
            probabilities: Weighted voting scores
        """
        # Get predictions from all models
        all_model_preds = []
        
        for i in range(self.n_models):
            preds, labels = self.get_predictions(i, loader)
            all_model_preds.append(preds)
        
        all_model_preds = np.array(all_model_preds)  # Shape: (n_models, n_samples)
        
        # Weighted voting
        # Convert predictions to {-1, 1} for AdaBoost
        weighted_votes = np.zeros(all_model_preds.shape[1])
        
        for i in range(self.n_models):
            vote = 2 * all_model_preds[i] - 1  # Convert {0,1} to {-1,1}
            weighted_votes += self.model_weights[i] * vote
        
        # Final predictions
        final_predictions = (weighted_votes > 0).astype(int)
        
        return final_predictions, weighted_votes, labels
    
    def evaluate(self, loader: DataLoader, set_name: str = "Test") -> dict:
        """Evaluate AdaBoost ensemble"""
        print(f"\n{'='*70}")
        print(f"Evaluating on {set_name} Set")
        print(f"{'='*70}")
        
        predictions, scores, labels = self.predict(loader)
        
        accuracy = accuracy_score(labels, predictions)
        
        print(f"\n{set_name} Accuracy: {accuracy*100:.2f}%")
        print(f"\nClassification Report:")
        print(classification_report(labels, predictions, 
                                   target_names=['Fake', 'Real'], 
                                   digits=4))
        print(f"{'='*70}\n")
        
        return {
            'accuracy': float(accuracy),
            'predictions': predictions,
            'scores': scores,
            'labels': labels
        }


# ================== Model Loading ==================

def load_pretrained_models(model_paths: List[str], device: torch.device) -> List[nn.Module]:
    """Load pre-trained models"""
    try:
        from model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError("Cannot import ResNet_50_pruned_hardfakevsreal")
    
    models = []
    print(f"\n{'='*70}")
    print(f"Loading {len(model_paths)} Pre-trained Models")
    print(f"{'='*70}")
    
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue
        
        print(f"\n[{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=checkpoint['masks'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  → Parameters: {param_count:,}")
            
            models.append(model)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue
    
    if len(models) == 0:
        raise ValueError("No models loaded successfully!")
    
    print(f"\n{'='*70}")
    print(f"Successfully loaded {len(models)} models")
    print(f"{'='*70}\n")
    
    return models


# ================== Main Function ==================

def main():
    parser = argparse.ArgumentParser(description="AdaBoost with Pre-trained Models")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'uadfV', 
                               'custom_genai', 'custom_genai_v2', 'real_fake_dataset'],
                       help='Dataset type')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    
    # Model parameters
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to pre-trained model checkpoints')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                       help='Names for each model')
    
    # AdaBoost parameters
    parser.add_argument('--n_iterations', type=int, default=None,
                       help='Number of AdaBoost iterations (default: number of models)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./adaboost_output',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths!")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"AdaBoost Ensemble with Pre-trained Models")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of models: {len(args.model_paths)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Normalization parameters for each model
    # Adjust these based on your models' training
    MEANS = [
        (0.5207, 0.4258, 0.3806),
        (0.4460, 0.3622, 0.3416),
        (0.4668, 0.3816, 0.3414)
    ]
    STDS = [
        (0.2490, 0.2239, 0.2212),
        (0.2057, 0.1849, 0.1761),
        (0.2410, 0.2161, 0.2081)
    ]
    
    # Extend if more models
    num_models = len(args.model_paths)
    if num_models > len(MEANS):
        MEANS = MEANS + [MEANS[-1]] * (num_models - len(MEANS))
        STDS = STDS + [STDS[-1]] * (num_models - len(STDS))
    
    # Load pre-trained models
    models = load_pretrained_models(args.model_paths, device)
    model_names = args.model_names[:len(models)]
    
    # Load datasets
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_type=args.dataset,
        is_distributed=False,
        seed=args.seed,
        is_main=True
    )
    
    # Evaluate individual models first
    print("\n" + "="*70)
    print("Individual Model Performance (Before AdaBoost)")
    print("="*70)
    
    individual_results = []
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\nEvaluating Model {i+1}: {name}")
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {name}"):
                images = images.to(device)
                labels = labels.to(device)
                
                # Normalize
                mean = torch.tensor(MEANS[i]).view(1, 3, 1, 1).to(device)
                std = torch.tensor(STDS[i]).view(1, 3, 1, 1).to(device)
                images_norm = (images - mean) / std
                
                # Predict
                outputs = model(images_norm)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                
                preds = (outputs.squeeze(1) > 0).long()
                correct += preds.eq(labels.long()).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        individual_results.append({'name': name, 'accuracy': accuracy})
        print(f"{name}: {accuracy:.2f}%")
    
    best_single = max(individual_results, key=lambda x: x['accuracy'])
    print(f"\nBest Single Model: {best_single['name']} → {best_single['accuracy']:.2f}%")
    print("="*70)
    
    # Initialize AdaBoost
    adaboost = AdaBoostClassifier(
        models=models,
        device=device,
        means=MEANS[:len(models)],
        stds=STDS[:len(models)],
        model_names=model_names
    )
    
    # Train AdaBoost
    adaboost.fit(train_loader, n_iterations=args.n_iterations)
    
    # Evaluate on validation set
    val_results = adaboost.evaluate(val_loader, set_name="Validation")
    
    # Evaluate on test set
    test_results = adaboost.evaluate(test_loader, set_name="Test")
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"Best Single Model: {best_single['accuracy']:.2f}%")
    print(f"AdaBoost Ensemble: {test_results['accuracy']*100:.2f}%")
    print(f"Improvement: {test_results['accuracy']*100 - best_single['accuracy']:+.2f}%")
    print("="*70 + "\n")
    
    # Save results
    final_results = {
        'individual_models': individual_results,
        'best_single_model': best_single,
        'adaboost': {
            'model_weights': adaboost.model_weights.tolist(),
            'model_names': model_names,
            'validation_accuracy': val_results['accuracy'],
            'test_accuracy': test_results['accuracy'],
            'improvement': test_results['accuracy'] - best_single['accuracy']/100
        },
        'config': vars(args)
    }
    
    results_path = os.path.join(args.save_dir, 'adaboost_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to: {results_path}")
    
    # Save AdaBoost weights
    weights_path = os.path.join(args.save_dir, 'adaboost_weights.pt')
    torch.save({
        'model_weights': adaboost.model_weights,
        'model_names': model_names,
        'means': MEANS[:len(models)],
        'stds': STDS[:len(models)],
        'test_accuracy': test_results['accuracy']
    }, weights_path)
    print(f"AdaBoost weights saved to: {weights_path}\n")
    
    # Plot ROC and F1 curves
    try:
        print("Generating ROC and F1 curves...")
        plot_roc_and_f1(
            adaboost,
            test_loader,
            device,
            args.save_dir,
            model_names,
            is_main=True
        )
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")


if __name__ == "__main__":
    main()
