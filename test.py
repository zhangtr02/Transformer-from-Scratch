import os
import torch
import sentencepiece as spm

from models.transformer import Transformer
from utils.config import load_config
from utils.data_utils import get_dataloaders
from utils.metrics import compute_bleu
from utils.log import setup_logger
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed
from utils.metrics import AverageMeter


def test():
    # Load config
    config = load_config()
    
    # Set random seed
    set_seed(42)
    
    # Setup logger
    logger = setup_logger(config['paths']['log_dir'], 'test')
    logger.info("Starting evaluation...")
    
    # Get dataloaders
    train_loader, _, test_loader = get_dataloaders(config)

    # Get vocabularies from dataset
    src_vocab = train_loader.dataset.src_vocab
    tgt_vocab = train_loader.dataset.tgt_vocab
    
    # Create reverse vocabularies for decoding
    tgt_idx2token = {v: k for k, v in tgt_vocab.items()}

    # Load SentencePiece model for decoding
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(config['data']['tgt_sp_model'])

    # Get special token indices
    pad_idx = tgt_vocab['<pad>']
    bos_idx = tgt_vocab['<bos>']
    eos_idx = tgt_vocab['<eos>']
    unk_idx = tgt_vocab['<unk>']
    special_ids = [pad_idx, bos_idx, eos_idx, unk_idx]
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        num_layers=config['model']['num_layers'],
        max_seq_length=config['model']['max_seq_length'],
        dropout=0.0  # No dropout during evaluation
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
    load_checkpoint(checkpoint_path, model)

    # Evaluation
    model.eval()
    test_bleu= 0.0
    test_bleu_meter = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_loader):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input, _ = tgt[:, :-1], tgt[:, 1:]
            
            # Create masks
            src_mask, _ = model.generate_mask(src, tgt_input)
            
            # Generate translation
            output = model.generate(src, src_mask, max_length=config['model']['max_seq_length'], start_token=bos_idx, end_token=eos_idx)
            
            # Convert indices to tokens
            references = [[tgt_idx2token[idx.item()] for idx in seq if idx.item() not in special_ids]
                         for seq in tgt]
            candidates = [[tgt_idx2token[idx.item()] for idx in seq if idx.item() not in special_ids]
                         for seq in output]
            
            # Decode BPE tokens to normal sentences
            decoded_references = [sp_en.decode(ref) for ref in references]
            decoded_candidates = [sp_en.decode(cand) for cand in candidates]
 
            # Compute BLEU score
            batch_bleu = compute_bleu(references, candidates)
            test_bleu_meter.update(batch_bleu)

            if batch_idx % 100 == 0:
                logger.info(f"Example reference: {decoded_references[0]}")
                logger.info(f"Example candidate: {decoded_candidates[0]}")
    
    test_bleu = test_bleu_meter.avg
    logger.info(f"Test BLEU score: {test_bleu:.2f}")


if __name__ == '__main__':
    test()