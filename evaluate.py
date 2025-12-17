#!/usr/bin/env python3
"""
Evaluation script for trained Fuzzy Hesitant Ensemble
"""

import os
import argparse
import torch
import warnings

from utils.seed import set_seed
from utils.ddp import setup_ddp, cleanup_ddp
from utils.evaluation import evaluate_ensemble_final_ddp
from models.ensemble import FuzzyHesitantEnsemble, load_pruned_models
from data.dataloaders import create_dataloaders_ddp

warnings.filterwarnings("ignore")


def main():
    # Setup
    SEED = 42
    set_seed(SEED)
    
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Fuzzy Hesitant Ensemble")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--dataset', type=str,
                       choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'],
                       required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_memberships', type=int, default=3)
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    
    args = parser.parse_args()
    
    if is_main:
        print("="*70)
        print("EVALUATING FUZZY HESITANT ENSEMBLE")
        print("="*70)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Dataset: {args.dataset}")
        print(f"Data directory: {args.data_dir}")
        print("="*70 + "\n")
    
    # Normalization parameters
    MEANS = [(0.5207, 0.4258, 0.3806), (0.4868, 0.3972, 0.3624), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2296, 0.2066, 0.2009), (0.2410, 0.2161, 0.2081)]
    
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]
    
    # Load models
    base_models = load_pruned_models(args.model_paths, device, rank)
    MODEL_NAMES = args.model_names[:len(base_models)]
    
    # Create ensemble
    ensemble = FuzzyHesitantEnsemble(
        base_models, MEANS, STDS,
        num_memberships=args.num_memberships,
        freeze_models=True
    ).to(device)
    
    # Load checkpoint
    if is_main:
        print(f"Loading checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ensemble.hesitant_fuzzy.load_state_dict(checkpoint['hesitant_state_dict'])
    
    if is_main:
        print("Checkpoint loaded successfully!\n")
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders_ddp(
        args.data_dir, args.batch_size, rank, world_size, 
        dataset_type=args.dataset
    )
    
    # Evaluate
    if is_main:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
    
    acc, weights, memberships, activations = evaluate_ensemble_final_ddp(
        ensemble, test_loader, device, "Test", MODEL_NAMES, rank
    )
    
    if is_main:
        print("\n" + "="*70)
        print(f"Final Test Accuracy: {acc:.3f}%")
        print("="*70)
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
