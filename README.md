# Transformer From Scratch
This repo contains my PyTorch implementation of the original transformer model from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). This project helped me gain a better understanding of the intricate workings of transformers and how they are trained and evaluated. I hope this repo can also help others enhance their own understanding. Feel free to explore the code, experiment with different configurations, and adapt it to suit your specific needs.


## Quickstart

1. Change the configuration for the transformer model and the training settings in `config.py` to your liking.
2. Run `train.py` to start training your transformer model.
3. Monitor your training progress by launching TensorBoard.
4. Run `evaluation.py` to evaluate your model's performance on the test set.


## Use Your Own Datasets
The [WMT 2014 English-German dataset](https://huggingface.co/datasets/wmt/wmt14) is used by default for training, validation and evaluation. If you wish to use your own datasets, you can go to `config.py` and change the `load_train()`, `load_valid()` and `load_test()` methods under the `MyDatasets` class. Make sure that these methods return a `datasets.IterableDataset` object.

For example, suppose you want to test your model on the [OPUS](https://huggingface.co/datasets/Helsinki-NLP/opus-100) dataset, simply change `load_test()` to:

```
def load_test():
    test_data = load_dataset('Helsinki-NLP/opus-100', 'de-en', split='train+valid+test', streaming=True)
    return test_data
```

If you want to train a model that translates a different pair of language, you should also change `MyDatasets` class variables `source_language` and `target_language` to the keys used to access the respective languages in the dataset.


## Use/Train Your Own Tokenizer
I provide a BPE tokenizer trained on the WMT 2014 English-German dataset (`tokenizer.json`), but if you want to use your own dataset, you should train a new tokenizer or use an existing one.

To train a new tokenizer on your dataset, you can change tokenizer's configuration under `TokenizerConfig` in the `config.py` file, and then run `tokenizer.py`, which trains a BPE tokenizer using your training and validation data.

If you wish to use an existing tokenizer, simply change the `load_tokenizer()` method under `TokenizerConfig`. Make sure it returns a `tokenizers.Tokenizer` object.

For example, if you wish to use the [Google T5](https://huggingface.co/google-t5/t5-base) tokenizer, simply change `load_tokenizer()` to:

```
def load_tokenizer():
    tokenizer = Tokenizer.from_pretrained('google-t5/t5-base')
    return tokenizer
```
