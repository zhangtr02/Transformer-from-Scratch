import os
import torch
import torch.optim as optim
import sentencepiece as spm

from models.transformer import Transformer
from utils.config import load_config
from utils.data_utils import get_dataloaders
from utils.loss import LabelSmoothing
from utils.scheduler import NoamScheduler
from utils.metrics import AverageMeter, compute_bleu
from utils.log import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plot import plot_training_curves
from utils.seed import set_seed

def train():
    # Load config
    config = load_config()
    
    # Set random seed
    set_seed(42)

    # Setup logger
    logger = setup_logger(config['paths']['log_dir'], 'train')
    logger.info("Starting training...")
    
    # Get dataloaders
    train_loader, valid_loader, _ = get_dataloaders(config)
    
    # Get vocabularies from dataset
    src_vocab = train_loader.dataset.src_vocab
    tgt_vocab = train_loader.dataset.tgt_vocab

    logger.info(f"src_vocab size: {len(src_vocab)}")
    logger.info(f"tgt_vocab size: {len(tgt_vocab)}")

    # Get special token indices
    pad_idx = tgt_vocab['<pad>']
    bos_idx = tgt_vocab['<bos>']
    eos_idx = tgt_vocab['<eos>']
    unk_idx = tgt_vocab['<unk>']
    special_ids = [pad_idx, bos_idx, eos_idx, unk_idx]
        
    # Create reverse vocabularies for decoding
    tgt_idx2token = {v: k for k, v in tgt_vocab.items()}
    
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
        dropout=config['model']['dropout']
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'],
                           betas=tuple(config['training']['betas']),
                           eps=float(config['training']['eps']))
    scheduler = NoamScheduler(
        optimizer,
        d_model=config['model']['d_model'],
        warmup_steps=config['training']['warmup_steps']
    )

    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'latest_model.pth')
    start_epoch = 0
    best_val_bleu = 0.0
    if os.path.exists(checkpoint_path):
        logger.info(f'Resuming from checkpoint: {checkpoint_path}')
        start_epoch, best_val_bleu = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        logger.info(f'Resumed at epoch {start_epoch}, best_val_bleu={best_val_bleu:.4f}')

    # Initialize loss function
    criterion = LabelSmoothing(smoothing=config['training']['label_smoothing'], ignore_index=pad_idx)

    # Load SentencePiece model for decoding
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(config['paths']['tgt_sp_model'])
    
    # Training loop
    train_losses = []
    val_bleus = []
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Training phase
        model.train()
        train_loss_meter = AverageMeter()
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            # Create masks
            src_mask, tgt_mask = model.generate_mask(src, tgt_input)
            
            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss_meter.update(loss.item())
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch+1}/{config["training"]["epochs"]} '
                          f'Batch: {batch_idx}/{len(train_loader)} '
                          f'Loss: {train_loss_meter.avg:.4f}')
                pred_tokens = output[0].argmax(dim=-1).tolist()
                tgt_tokens = tgt_output[0].tolist()
                logger.info(f'Pred tokens: {pred_tokens[:10]}')
                logger.info(f'True tokens: {tgt_tokens[:10]}')
                
        train_losses.append(train_loss_meter.avg)
        
        # Validation phase
        model.eval()
        val_bleu_meter = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(valid_loader):
                src, tgt = src.to(device), tgt.to(device)

                tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                
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
                val_bleu_meter.update(batch_bleu)
                if batch_idx % 100 == 0:
                    logger.info(f'Example reference: {decoded_references[0]}')
                    logger.info(f'Example candidate: {decoded_candidates[0]}')
        
        val_bleus.append(val_bleu_meter.avg)
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch+1, best_val_bleu, config['paths']['checkpoint_dir'], 'latest_model.pth')

        # Save best checkpoint
        if val_bleu_meter.avg > best_val_bleu:
            best_val_bleu = val_bleu_meter.avg
            save_checkpoint(model, optimizer, scheduler, epoch+1, best_val_bleu, config['paths']['checkpoint_dir'], 'best_model.pth')
        
        logger.info(f'Epoch: {epoch+1}/{config["training"]["epochs"]} '
                   f'Train Loss: {train_loss_meter.avg:.4f} '
                   f'BLEU Score: {val_bleu_meter.avg:.2f}')

    logger.info("Training completed!")

    # Plot training curves
    plot_training_curves(train_losses, val_bleus, config['paths']['figure_dir'])

if __name__ == '__main__':
    train()