import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import json
from datasets import load_from_disk
import sentencepiece as spm

SPECIAL_TOKENS = ['<pad>', '<bos>', '<eos>', '<unk>']

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_sp, tgt_sp, src_vocab, tgt_vocab, max_length):
        self.dataset = dataset
        self.src_sp = src_sp
        self.tgt_sp = tgt_sp
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_line = item['translation']['de']
        tgt_line = item['translation']['en']

        # BPE tokenization
        src_tokens = self.src_sp.encode(src_line, out_type=str)
        tgt_tokens = self.tgt_sp.encode(tgt_line, out_type=str)

        # Convert tokens to indices
        src_indices = [self.src_vocab.get(token, self.src_vocab['<unk>']) for token in src_tokens]
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab['<unk>']) for token in tgt_tokens]

        # Add special tokens
        src_indices = [self.src_vocab['<bos>']] + src_indices + [self.src_vocab['<eos>']]
        tgt_indices = [self.tgt_vocab['<bos>']] + tgt_indices + [self.tgt_vocab['<eos>']]

        # Truncate and pad
        src_indices = src_indices[:self.max_length]
        tgt_indices = tgt_indices[:self.max_length]
        src_indices += [self.src_vocab['<pad>']] * (self.max_length - len(src_indices))
        tgt_indices += [self.tgt_vocab['<pad>']] * (self.max_length - len(tgt_indices))

        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def build_vocab_from_sp(sp_model, extra_tokens=SPECIAL_TOKENS):
    vocab = {}
    # Add extra tokens first
    for idx, token in enumerate(extra_tokens):
        vocab[token] = idx

    # Add tokens from the SentencePiece model
    idx = len(extra_tokens)
    for i in range(sp_model.get_piece_size()):
        token = sp_model.id_to_piece(i)
        if token not in vocab:
            vocab[token] = idx
            idx += 1   
    return vocab

def save_vocab(vocab: Dict[str, int], save_path: str) -> None:
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(load_path: str) -> Dict[str, int]:
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load datasets
    train_dataset = load_from_disk(config['paths']['train_path'])
    valid_dataset = load_from_disk(config['paths']['valid_path'])
    test_dataset = load_from_disk(config['paths']['test_path'])

    # Load BPE models
    src_sp = spm.SentencePieceProcessor()
    src_sp.load(config['paths']['src_sp_model'])
    tgt_sp = spm.SentencePieceProcessor()
    tgt_sp.load(config['paths']['tgt_sp_model'])

    # Build or load vocabularies
    os.makedirs(config['paths']['vocabularies_dir'], exist_ok=True)
    src_vocab_path = os.path.join(config['paths']['vocabularies_dir'], 'src_vocab.json')
    tgt_vocab_path = os.path.join(config['paths']['vocabularies_dir'], 'tgt_vocab.json')

    if not os.path.exists(src_vocab_path):
        src_vocab = build_vocab_from_sp(src_sp)
        save_vocab(src_vocab, src_vocab_path)
    else:
        src_vocab = load_vocab(src_vocab_path)

    if not os.path.exists(tgt_vocab_path):
        tgt_vocab = build_vocab_from_sp(tgt_sp)
        save_vocab(tgt_vocab, tgt_vocab_path)
    else:
        tgt_vocab = load_vocab(tgt_vocab_path)

    # Create datasets
    train_dataset = TranslationDataset(
        train_dataset,
        src_sp,
        tgt_sp,
        src_vocab,
        tgt_vocab,
        config['model']['max_seq_length']
    )
    valid_dataset = TranslationDataset(
        valid_dataset,
        src_sp,
        tgt_sp,
        src_vocab,
        tgt_vocab,
        config['model']['max_seq_length']
    )
    test_dataset = TranslationDataset(
        test_dataset,
        src_sp,
        tgt_sp,
        src_vocab,
        tgt_vocab,
        config['model']['max_seq_length']
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )

    return train_loader, valid_loader, test_loader