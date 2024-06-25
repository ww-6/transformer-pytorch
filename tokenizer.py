from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import TokenizerConfig, MyDatasets
from itertools import chain
from typing import Iterator


special_tokens = [
    TokenizerConfig.pad_token,
    TokenizerConfig.start_token,
    TokenizerConfig.end_token,
]


train = MyDatasets.load_train().iter(TokenizerConfig.batch_size)
valid = MyDatasets.load_valid().iter(TokenizerConfig.batch_size)
train_valid = chain(train, valid)


def batch_iterator() -> Iterator[list[str]]:
    for batch in train_valid:
        yield [sentence for pair in batch["translation"] for sentence in pair.values()]


trainer = BpeTrainer(
    vocab_size=TokenizerConfig.vocab_size, special_tokens=special_tokens
)

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(iterator=batch_iterator(), trainer=trainer)

tokenizer.save(TokenizerConfig.filename)
