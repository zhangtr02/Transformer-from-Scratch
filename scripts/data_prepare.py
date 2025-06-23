import os
import sentencepiece as spm
from datasets import load_dataset

def download_wmt14():
    """Download WMT14 German-English dataset"""
    dataset = load_dataset('wmt14', 'de-en')
    dataset.save_to_disk('data/wmt14_de_en')
    return dataset

def export_parallel_corpus(dataset, split='train', save_dir='data/wmt14_de_en'):
    """Export parallel corpus as raw text files for BPE training."""
    with open(f'{save_dir}/{split}.de', 'w', encoding='utf-8') as f_de,\
        open(f'{save_dir}/{split}.en', 'w', encoding='utf-8') as f_en:
        for item in dataset[split]:
            f_de.write(item['translation']['de'].strip() + '\n')
            f_en.write(item['translation']['en'].strip() + '\n')

def train_sentencepiece_bpe(input_path, model_prefix, vocab_size=32000, character_coverage=1.0):
    """Train a SentencePiece BPE model."""
    spm.SentencePieceTrainer.Train(
        f'--input={input_path} --model_prefix={model_prefix} --vocab_size={vocab_size} '
        f'--character_coverage={character_coverage} --model_type=bpe'
    )

if __name__ == '__main__':
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    # Download datasets
    dataset = download_wmt14()

    # Export parallel corpus for train split
    export_parallel_corpus(dataset, split='train', save_dir='data/wmt14_de_en')
    print(f'Exported train split to raw text files.')

    # Train BPE models
    os.makedirs('bpe_models', exist_ok=True)
    train_sentencepiece_bpe(
        input_path='data/wmt14_de_en/train.en',
        model_prefix='bpe_models/bpe_en'
    )
    train_sentencepiece_bpe(
        input_path='data/wmt14_de_en/train.de',
        model_prefix='bpe_models/bpe_de'
    )