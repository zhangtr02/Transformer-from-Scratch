import matplotlib.pyplot as plt
import os
from typing import List

def plot_training_curves(train_losses: List[float], val_bleus: List[float], save_dir: str):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_bleus: List of validation BLEU scores
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
        
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()
    
    # Plot BLEU score curve
    plt.figure(figsize=(10, 5))
    plt.plot(val_bleus, label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'bleu_score_curve.png'))
    plt.close()