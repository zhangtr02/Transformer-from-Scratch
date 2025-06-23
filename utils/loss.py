import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = pred.size()

        # Flatten tensors for easier masking
        pred = pred.reshape(-1, vocab_size)     # (batch_size * seq_len, vocab_size)
        target = target.reshape(-1)             # (batch_size * seq_len)

        # Create mask for valid positions
        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]
        
        # Create smoothed one-hot targets
        with torch.no_grad():
            true_dist = F.one_hot(target, num_classes=vocab_size).float()
        
            # Apply label smoothing
            true_dist = true_dist * (1 - self.smoothing) + self.smoothing / vocab_size
        
        # Compute cross entropy loss
        loss = torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1)
        return loss.mean()