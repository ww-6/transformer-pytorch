import torch
from config import TokenizerConfig, ModelConfig, MyDatasets
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer
from torch import Tensor



def prepare_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    '''Add special tokens, enable padding, enable truncation.'''

    special_tokens = [
        TokenizerConfig.pad_token,
        TokenizerConfig.start_token,
        TokenizerConfig.end_token,
    ]

    tokenizer.add_special_tokens(special_tokens)

    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(TokenizerConfig.pad_token),
        pad_token=TokenizerConfig.pad_token,
    )

    tokenizer.enable_truncation(max_length=ModelConfig.max_seq_len)

    return tokenizer



def get_sentences(batch: dict[str, list], language='en') -> list[str]:
    '''Get the sentences of one language from a batch that contains two or more languages.

    Args:
        batch: Should be in the following form
            {'id': [int, ...], 'translation': [{'language1': str, 'language2': str}, ...]}
        language: The key used to access the sentence from this language.

    Returns:
        A list of sentences in one language. 
    '''
    return [x[language] for x in batch['translation']]



def get_token_ids(batch) -> list[list[int]]:
    '''Get token ids from the output of Tokenizer.encode_batch()'''
    return [x.ids for x in batch]



def get_padding_masks(batch) -> list[list[int]]:
    '''Get padding masks from the output of Tokenizer.encode_batch()'''
    return [x.attention_mask for x in batch]



def tokenize(
    sentences: list[str], 
    tokenizer: Tokenizer, 
    post_processor=None, 
    device=None
) -> tuple[Tensor, Tensor]:
    '''Tokenize the batch and perform optional post processing.

    Args:
        batch: A list of sentences.
        tokenizer: An instance of tokenizers.Tokenizer.
        post_processor: An optional post processor from tokenizers.processors.
        device: Which device the output tensors should be moved to.

    Returns:
        A tuple containing:
            - A tensor of token ids, shape (batch size, T).
            - A tensor of padding masks, shape (batch size, T).
        where T is the number of tokens of the longest sentence in the batch.
    '''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if post_processor is not None:
        tokenizer.post_processor = post_processor

    sentences = tokenizer.encode_batch(sentences)
    token_ids = get_token_ids(sentences)
    padding_masks = get_padding_masks(sentences)

    token_ids = torch.tensor(token_ids, device=device)
    padding_masks = torch.tensor(padding_masks, device=device)

    return token_ids, padding_masks



def tokenize_source(sentences: list[str], tokenizer: Tokenizer, device=None) -> tuple[Tensor, Tensor]:
    '''Tokenize the source language and add an end of sentence special token at the end.

    Args:
        sentences: A list of sentences.
        tokenizer: An instance of tokenizers.Tokenizer.
        device: Which device the output tensors should be moved to.
    
    Returns:
        A tuple containing:
            - A tensor of token ids, shape (batch size, T).
            - A tensor of padding masks, shape (batch size, T).
        where T is the number of tokens of the longest sentence in the batch.
    '''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    post_processor = TemplateProcessing(
        single='$A ' + TokenizerConfig.end_token,
        special_tokens=[
            (
                TokenizerConfig.end_token,
                tokenizer.token_to_id(TokenizerConfig.end_token),
            )
        ],
    )

    token_ids, padding_masks = tokenize(sentences, tokenizer, post_processor, device)
    return token_ids, padding_masks



def tokenize_target(sentences: list[str], tokenizer: Tokenizer, device=None) -> tuple[Tensor, Tensor]:
    '''Tokenize the target language and add a start of sentence token and an end of sentence token.
    
    Args:
        sentences: A list of sentences.
        tokenizer: An instance of tokenizers.Tokenizer.
        device: Which device the output tensors should be moved to.

    Returns:
        A tuple containing:
            - A tensor of token ids, shape (batch size, T + 1).
            - A tensor of padding masks, shape (batch size, T).
        where T is the number of tokens of the longest sentence in the batch.
    '''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    post_processor = TemplateProcessing(
        single=' '.join([
            TokenizerConfig.start_token, 
            '$A', 
            TokenizerConfig.end_token, 
            TokenizerConfig.pad_token
        ]),
        special_tokens=[
            (
                TokenizerConfig.start_token,
                tokenizer.token_to_id(TokenizerConfig.start_token),
            ),
            (
                TokenizerConfig.end_token,
                tokenizer.token_to_id(TokenizerConfig.end_token),
            ),
            (
                TokenizerConfig.pad_token,
                tokenizer.token_to_id(TokenizerConfig.pad_token)
            )
        ],
    )

    token_ids, padding_masks = tokenize(sentences, tokenizer, post_processor, device)

    return token_ids, padding_masks[:, 1:]



def process_batch(
        batch: dict[str, list], 
        tokenizer: Tokenizer, 
        device=None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''Tokenize a batch from a translation dataset.
    The tokenized source language includes the end of sentence token.
    The tokenized target language includes the start of sentence token,
    and the end of sentence token.
    
    Sentences from the source language are padded to the same length as 
    the longest sentence in the batch.

    Sentences from the target language are padded to T_t + 1, where
    T_t is  the number of tokens of the longest sentence in the batch.
    
    
    Args:
        batch: Should be in the following form
            {'id': [int, ...], 'translation': [{'language1': str, 'language2': str}, ...]}
        tokenizer: An instance of tokenizers.Tokenizer
        device: Which device the output tensors should be moved to.

        
    Returns:
        A tuple containing:
            - A tensor of source language token ids, shape (batch size, T_s).
            - A tensor of source language padding masks, shape (batch size, T_s).
            - A tensor of target language token ids, shape (batch size, T_t + 1).
            - A tensor of target language padding masks, shape (batch size, T_t).
        where T_s is the number of tokens of the source language's longest sentence,
        and T_t is the number of tokens of the target language's longest sentence.
    '''


    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_sentences = get_sentences(batch, MyDatasets.source_language)
    target_sentences = get_sentences(batch, MyDatasets.target_language)

    source_token_ids, source_padding_masks = tokenize_source(
        source_sentences, tokenizer, device
    )
    target_token_ids, target_padding_masks = tokenize_target(
        target_sentences, tokenizer, device
    )

    return source_token_ids, source_padding_masks, target_token_ids, target_padding_masks

