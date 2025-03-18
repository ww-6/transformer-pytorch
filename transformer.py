import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from modules import PositionalEncoding, EncoderBlock, DecoderBlock
from torch import Tensor
from typing import Optional
from utils import tokenize_source
import math


class Transformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        vocab_size: int,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        bias: bool = True,
        layer_norm_eps: float = 10**-5,
        padding_idx: Optional[int] = None
    ) -> None:

        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.embd_scale = math.sqrt(d_model)
        self.pe = PositionalEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.linear.weight = self.embedding.weight
        
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    max_seq_len,
                    d_model, 
                    n_heads, 
                    feedforward_dim, 
                    dropout, 
                    bias, 
                    layer_norm_eps
                ) for _ in range(n_encoder_layers)
            ]
        )
        
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    max_seq_len,
                    d_model, 
                    n_heads, 
                    feedforward_dim, 
                    dropout, 
                    bias, 
                    layer_norm_eps
                ) for _ in range(n_decoder_layers)
            ]
        )

        self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def encode(self, source: Tensor, source_padding_masks: Optional[Tensor] = None) -> Tensor:
        encoder_output = self.embedding(source) * self.embd_scale
        encoder_output = self.pe(encoder_output, source_padding_masks)
        encoder_output = self.dropout(encoder_output)

        for module in self.encoder:
            encoder_output = module(encoder_output, source_padding_masks)
        return encoder_output


    def decode(
        self, 
        target: Tensor, 
        encoder_output: Tensor, 
        source_padding_masks: Optional[Tensor] = None, 
        target_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        decoder_output = self.embedding(target) * self.embd_scale
        decoder_output = self.pe(decoder_output, target_padding_masks)
        decoder_output = self.dropout(decoder_output)

        for module in self.decoder:
            decoder_output = module(decoder_output, encoder_output, source_padding_masks, target_padding_masks)
        
        return decoder_output
    

    def forward(
        self, 
        source: Tensor, 
        target: Tensor, 
        source_padding_masks: Optional[Tensor] = None, 
        target_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        encoder_output = self.encode(source, source_padding_masks)
        decoder_output = self.decode(target, encoder_output, source_padding_masks, target_padding_masks)
        
        # (B, T, d_model) x (d_model, vocab_size) -> (B, T, vocab_size)
        logits = self.linear(decoder_output)
        return logits


    @torch.no_grad()
    def translate(
        self, 
        sentences: list[str], 
        tokenizer: Tokenizer, 
        start_token: str = '<BOS>', 
        end_token: str = '<EOS>', 
        max_tokens: Optional[int] = None
    ) -> list[str]:
        
        '''Given a list of sentences in the source language, translate them to the target language
        using greedy search.

        Args:
            sentences: A list of sentences in the source language.
            tokenizer: The same tokenizer used for the training of the model.
            start_token: The start of the sentence token used during training.
            end_token: The end of the sentence token used during training.
            max_tokens: The maximum number of tokens the model is allowed to generate 
                for each translation. Default: `max_seq_len`

        Returns:
            A list of translated sentences in the target language.
        '''

        if max_tokens is None:
            max_tokens = self.max_seq_len
        else:
            assert max_tokens <= self.max_seq_len
        
        self.eval()
        device = self.embedding.weight.device
        B = len(sentences)
        end_token_id = tokenizer.token_to_id(end_token)
        pad_token_id = tokenizer.padding['pad_id']

        source_token_ids, source_padding_masks = tokenize_source(sentences, tokenizer, device)
        encoder_output = self.encode(source_token_ids, source_padding_masks)
        translation = torch.full((B, 1), tokenizer.token_to_id(start_token), dtype=torch.long, device=device)
        new_translation = torch.full((B, 1), pad_token_id, dtype=torch.long, device=device)

        # Indicate which sentence has finished translation.
        end_masks = torch.full((B,), False, device=device)

        for i in range(max_tokens):
            decoder_output = self.decode(
                translation[~end_masks, :], 
                encoder_output[~end_masks, :], 
                source_padding_masks[~end_masks, :],
            )
            logits = self.linear(decoder_output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            new_token_ids = torch.argmax(probs, dim=-1)
            new_translation[~end_masks, 0] = new_token_ids

            translation = torch.concat((translation, new_translation), dim=-1)
            end_masks[~end_masks] = (new_token_ids == end_token_id)

            if all(end_masks):
                break

            new_translation[:] = pad_token_id
        
        translation = tokenizer.decode_batch(translation.tolist(), skip_special_tokens=True)

        return translation