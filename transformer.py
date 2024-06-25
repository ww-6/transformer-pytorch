import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from modules import PositionalEncoding, EncoderBlock, DecoderBlock
from torch import Tensor
from typing import Optional
from utils import tokenize_source


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
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pe = PositionalEncoding(max_seq_len, d_model)
        
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
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
                    d_model, 
                    n_heads, 
                    feedforward_dim, 
                    dropout, 
                    bias, 
                    layer_norm_eps
                ) for _ in range(n_decoder_layers)
            ]
        )


    def encode(self, source: Tensor, source_padding_masks: Optional[Tensor] = None) -> Tensor:
        output = self.embedding(source) * torch.sqrt(torch.tensor(self.d_model))
        output = self.pe(output, source_padding_masks)
        for module in self.encoder:
            output = module(output, source_padding_masks)
        return output


    def decode(
        self, 
        target: Tensor, 
        encoder_output: Tensor, 
        source_padding_masks: Optional[Tensor] = None, 
        target_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        decoder_output = self.embedding(target) * torch.sqrt(torch.tensor(self.d_model))
        decoder_output = self.pe(decoder_output, target_padding_masks)

        for module in self.decoder:
            decoder_output = module(decoder_output, encoder_output, source_padding_masks, target_padding_masks)
        
        # (B, T, d_model) x (1, d_model, vocab_size) -> (B, T, vocab_size)
        logits = decoder_output @ self.embedding.weight.T.unsqueeze(0)
        return logits
    

    def forward(
        self, 
        source: Tensor, 
        target: Tensor, 
        source_padding_masks: Optional[Tensor] = None, 
        target_padding_masks: Optional[Tensor] = None
    ) -> Tensor:
        
        encoder_output = self.encode(source, source_padding_masks)
        logits = self.decode(target, encoder_output, source_padding_masks, target_padding_masks)
        return logits


    @torch.no_grad()
    def translate(
        self, 
        sentences: list[str], 
        tokenizer: Tokenizer, 
        start_token: str = '<BOS>', 
        end_token: str = '<EOS>', 
        max_tokens: int = 100
    ) -> list[str]:
        
        '''Given a list of sentences in the source language, translate them to the target language
        using greedy search.

        Args:
            sentences: A list of sentences in the source language.
            tokenizer: The same tokenizer used for the training of the model.
            start_token: The start of the sentence token used during training.
            end_token: The end of the sentence token used during training.
            max_tokens: The maximum number of tokens the model is allowed to generate 
                for each translation.

        Returns:
            A list of translated sentences in the target language.
        '''
        
        self.eval()
        device = self.embedding.weight.device
        B = len(sentences)

        source_token_ids, source_padding_masks = tokenize_source(sentences, tokenizer, device)
        encoder_output = self.encode(source_token_ids, source_padding_masks)
        translation = torch.full((B, 1), tokenizer.token_to_id(start_token), device=device)

        end_token_id = tokenizer.token_to_id(end_token)
        pad_token_id = tokenizer.padding['pad_id']

        # Indicate which sentence has finished translation.
        end_masks = torch.full((B,), False)

        for i in range(max_tokens+1):
            logits = self.decode(
                translation[~end_masks, :], 
                encoder_output[~end_masks, :], 
                source_padding_masks[~end_masks, :],
            )
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token_ids = torch.argmax(probs, dim=-1)
            new_translation = torch.zeros((B, 1), dtype=int, device=device)
            new_translation[~end_masks, :] = new_token_ids.unsqueeze(-1)
            new_translation[end_masks, :] = pad_token_id
            
            translation = torch.concat((translation, new_translation), dim=-1)

            new_finished_translation = (new_token_ids == end_token_id).cpu()

            if any(new_finished_translation):
                end_masks[~end_masks] = new_finished_translation

            if all(end_masks):
                break
        
        translation = tokenizer.decode_batch(translation.tolist(), skip_special_tokens=True)

        return translation