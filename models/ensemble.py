import os
import torch
import torch.nn as nn
from typing import List, Tuple

from .hesitant_fuzzy import HesitantFuzzyMembership, MultiModelNormalization


class FuzzyHesitantEnsemble(nn.Module):
    """
    Fuzzy Hesitant Ensemble model that combines multiple base models
    using hesitant fuzzy sets for dynamic model selection and weighting
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        means: List[Tuple[float]], 
        stds: List[Tuple[float]], 
        num_memberships: int = 3, 
        freeze_models: bool = True,
        cum_weight_threshold: float = 0.9, 
        hesitancy_threshold: float = 0.2
    ):
        """
        Args:
            models: List of pre-trained base models
            means: List of normalization means for each model
            stds: List of normalization stds for each model
            num_memberships: Number of membership values per model
            freeze_models: Whether to freeze base model parameters
            cum_weight_threshold: Cumulative weight threshold for model selection
            hesitancy_threshold: Threshold for high hesitancy detection
        """
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)
        self.hesitant_fuzzy = HesitantFuzzyMembership(
            input_dim=128,
            num_models=self.num_models,
            num_memberships=num_memberships
        )
        self.cum_weight_threshold = cum_weight_threshold
        self.hesitancy_threshold = hesitancy_threshold
       
        if freeze_models:
            for model in self.models:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
   
    def forward(self, x: torch.Tensor, return_details: bool = False):
        """
        Forward pass through the ensemble
        
        Args:
            x: Input tensor (B, C, H, W)
            return_details: Whether to return detailed information
            
        Returns:
            If return_details=False:
                final_output: Weighted ensemble output
                final_weights: Model weights used
            If return_details=True:
                final_output: Weighted ensemble output
                final_weights: Model weights used
                all_memberships: All membership values
                outputs: Individual model outputs
        """
        # Generate fuzzy membership weights
        final_weights, all_memberships = self.hesitant_fuzzy(x)
       
        # Calculate hesitancy (variance across memberships)
        hesitancy = all_memberships.var(dim=2)  # (B, num_models)
        avg_hesitancy = hesitancy.mean(dim=1)  # (B,)
       
        # Initialize selection mask
        mask = torch.ones_like(final_weights)
        high_hesitancy_mask = (avg_hesitancy > self.hesitancy_threshold).unsqueeze(1)
       
        # Apply cumulative weight threshold for model selection
        sorted_weights, sorted_indices = torch.sort(final_weights, dim=1, descending=True)
        cum_weights = torch.cumsum(sorted_weights, dim=1)
       
        for b in range(x.size(0)):
            if high_hesitancy_mask[b]:
                # High hesitancy: use all models
                continue
            
            # Low hesitancy: select top models until threshold
            active_count = torch.sum(cum_weights[b] < self.cum_weight_threshold) + 1
            top_indices = sorted_indices[b, :active_count]
            sample_mask = torch.zeros(self.num_models, device=x.device)
            sample_mask[top_indices] = 1.0
            mask[b] = sample_mask
       
        # Apply mask and renormalize weights
        final_weights = final_weights * mask
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-8)
       
        # Initialize output tensor
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
       
        # Collect active model indices across all samples
        active_model_indices = set()
        for b in range(x.size(0)):
            active_model_indices.update(
                torch.nonzero(final_weights[b] > 0).squeeze(-1).cpu().tolist()
            )
       
        # Run only active models
        for i in list(active_model_indices):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            outputs[:, i] = out
       
        # Weighted combination
        final_output = (outputs * final_weights.unsqueeze(-1)).sum(dim=1)
       
        if return_details:
            return final_output, final_weights, all_memberships, outputs
        return final_output, final_weights


def load_pruned_models(
    model_paths: List[str], 
    device: torch.device, 
    rank: int
) -> List[nn.Module]:
    """
    Load pruned models from checkpoints
    
    Args:
        model_paths: List of paths to model checkpoints
        device: Device to load models on
        rank: Current process rank (for distributed training)
        
    Returns:
        List of loaded models
    """
    try:
        from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
    except ImportError:
        raise ImportError(
            "Cannot import ResNet_50_pruned_hardfakevsreal. "
            "Ensure model.pruned_model.ResNet_pruned is available."
        )
   
    models = []
    if rank == 0:
        print(f"Loading {len(model_paths)} pruned models...")
 
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if rank == 0:
                print(f" [WARNING] File not found: {path}")
            continue
     
        if rank == 0:
            print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
     
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ResNet_50_pruned_hardfakevsreal(masks=ckpt['masks'])
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()
         
            if rank == 0:
                param_count = sum(p.numel() for p in model.parameters())
                print(f" → Parameters: {param_count:,}")
         
            models.append(model)
        except Exception as e:
            if rank == 0:
                print(f" [ERROR] Failed to load {path}: {e}")
            continue
 
    if len(models) == 0:
        raise ValueError("No models loaded!")
 
    if rank == 0:
        print(f"All {len(models)} models loaded!\n")
 
    return models
