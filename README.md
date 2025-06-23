# Transformer-from-Scratch

A PyTorch implementation of the Transformer model, faithfully reproducing the architecture and training procedure described in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).

## Project Structure

```
Transformer-from-Scratch/
├── bpe_models/         # SentencePiece BPE models
├── checkpoints/        # Model checkpoints
├── config/             # Configuration files
│   └── config.yaml
├── data/               # Raw and processed data
├── figures/            # Training/validation figures
├── logs/               # Training and evaluation logs
├── vocabularies/       # Vocabulary files
├── models/
│   └── transformer.py
├── scripts/
│   └── data_prepare.py
├── utils/
│   ├── data_utils.py
│   ├── scheduler.py
│   ├── loss.py
│   ├── metrics.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── log.py
│   ├── seed.py
│   └── plot.py
├── train.py
├── test.py
└── README.md
```

## Getting Started

### 1. Prepare Data and Train BPE Models

Download the WMT14 German-English dataset and train SentencePiece BPE models:

```bash
python scripts/data_prepare.py
```

### 2. Train the Transformer Model

Train the Transformer model from scratch:

```bash
python train.py
```

### 3. Evaluate the Model

Test the trained model and compute the BLEU score:

```bash
python test.py
```

## Features

- Encoder-Decoder Transformer architecture
- Multi-head self-attention and cross-attention
- Position-wise feed-forward networks
- Positional encoding
- Label smoothing loss
- Noam learning rate scheduler
- SentencePiece BPE tokenization
- Greedy decoding for sequence generation
- BLEU score evaluation
- Training/validation curve visualization
- Checkpoint saving and loading

## Requirements

- Python 3.8+
- PyTorch
- sentencepiece
- datasets
- tqdm

Install dependencies with:

```bash
pip install torch sentencepiece datasets tqdm
```

## Configuration

You can modify hyperparameters and paths in `config.yaml`.

## Citation

If you use this code, please cite the original paper:

> Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.