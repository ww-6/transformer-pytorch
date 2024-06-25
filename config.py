from dataclasses import dataclass
from datasets import load_dataset, IterableDataset
from tokenizers import Tokenizer


@dataclass
class TokenizerConfig:
    pad_token: str = '<PAD>'
    start_token: str = '<BOS>'
    end_token: str = '<EOS>'
    batch_size: int = 100
    filename: str = 'tokenizer.json'
    vocab_size: int = 37000

    @staticmethod
    def load_tokenizer() -> Tokenizer:
        tokenizer = Tokenizer.from_file(TokenizerConfig.filename)
        return tokenizer


@dataclass
class ModelConfig:
    vocab_size: int
    padding_idx: int
    max_seq_len: int = 256
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.1
    bias: bool = True
    layer_norm_eps: float = 10**-5


@dataclass
class OptimizerConfig:
    adam_betas: tuple = (0.9, 0.98)
    adam_eps: float = 10**-9
    warmup_steps: int = 4000


@dataclass
class TrainConfig:
    save_directory: str = 'checkpoints'
    resume: bool = False
    checkpoint_file: str = 'checkpoints/best_model.pt'
    device: str = 'cuda'
    max_steps: int = 100_000
    batch_size: int = 32
    label_smoothing: float = 0.1
    eval_interval: int = 1000
    log_interval: int = 1000
    save_interval: int = 0
    save_every_epoch: bool = False
    early_stopping: bool = True
    early_stop_thresh: int = 10_000
    buffer_size: int = 10_000
    seed: int = 123


@dataclass
class EvaluationConfig:
    checkpoint_file: str = 'checkpoints/best_model.pt'
    device: str = 'cuda'
    max_tokens: int = 256
    batch_size: int = 64


@dataclass
class MyDatasets:   
    source_language: str = 'en'
    target_language: str = 'de'

    @staticmethod
    def load_train() -> IterableDataset:
        train_data = load_dataset('wmt/wmt14', 'de-en', split='train', streaming=True)
        return train_data

    @staticmethod
    def load_valid() -> IterableDataset:
        valid_data = load_dataset('wmt/wmt14', 'de-en', split='validation', streaming=True)
        return valid_data

    @staticmethod
    def load_test() -> IterableDataset:
        test_data = load_dataset('wmt/wmt14', 'de-en', split='test', streaming=True)
        return test_data