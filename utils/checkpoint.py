import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_bleu, checkpoint_dir, filename):
    """
    Save model, optimizer, scheduler, epoch, and best BLEU to checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_bleu': best_val_bleu
    }, filepath)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load checkpoint and restore model, optimizer, scheduler, epoch, and best BLEU.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_val_bleu = checkpoint.get('best_val_bleu', 0.0)
    return epoch, best_val_bleu