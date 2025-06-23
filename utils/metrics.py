import sacrebleu
from typing import List

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_bleu(references: List[List[str]], candidates: List[List[str]]) -> float:
    """
    Compute BLEU score using sacrebleu.
    
    Args:
        references: List of reference sentences (each a list of tokens)
        candidates: List of candidate sentences (each a list of tokens)
        
    Returns:
        BLEU score (float)
    """
    references_str = [[' '.join(ref) for ref in references]]
    candidates_str = [''.join(c) for c in candidates]

    bleu = sacrebleu.corpus_bleu(candidates_str, references_str, lowercase=True)
    return bleu.score