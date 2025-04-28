from typing import List, Union, Literal
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,        # PyTorch
    TFAutoModel       # TensorFlow
)

import torch
import tensorflow as tf


class EncoderWithChunks:
    """
    Generic encoder that chunks long texts to respect the underlying model's
    token limit and returns one embedding per chunk (CLS or mean pooled).

    Parameters
    ----------
    model_name : str
        Any model hosted on HuggingFace (e.g. "bert-base-uncased",
        "roberta-base", "allenai/mpnet-base", "sentence-transformers/all-MiniLM-L6-v2").
    max_tokens : int | None
        Maximum sequence length including special tokens.  If None, uses
        tokenizer.model_max_length.
    framework : Literal["pt", "tf"]
        "pt" for PyTorch (default) or "tf" for TensorFlow.
    pooling : Literal["cls", "mean"]
        How to pool each chunk.  "cls" uses token 0, "mean" averages valid tokens.
    device : str | None
        Manually pick device for PyTorch ("cpu", "cuda").  Ignored for TensorFlow.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_tokens: int | None = None,
        framework: Literal["pt", "tf"] = "pt",
        pooling: Literal["cls", "mean"] = "cls",
        device: str | None = None,
    ):
        # --- load model & tokenizer ---------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.framework = framework
        self.pooling = pooling

        if framework == "pt":
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:  # TensorFlow
            self.device = None
            self.model = TFAutoModel.from_pretrained(model_name, from_pt=True)

        self.max_tokens = max_tokens or self.tokenizer.model_max_length

        # Reserve two slots for special tokens
        self.chunk_size = self.max_tokens - 2
        self.stride = self.chunk_size # no overlap
        # self.stride = self.chunk_size // 2  # 50 % overlap


    def _split_into_chunks(self, text: str) -> List[List[int]]:
        """Return list of token-ID lists each ≤ chunk_size."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, len(tokens), self.stride): 
            chunk_tokens = tokens[start:start + self.chunk_size]
            chunks.append(chunk_tokens)
            if start + self.chunk_size >= len(tokens):
                break
        return chunks


    def _prepare_inputs(self, chunk_ids: List[int]):
        """Add special tokens, pad, and create attention mask."""
        ids = (
            [self.tokenizer.cls_token_id]
            + chunk_ids[: self.chunk_size]
            + [self.tokenizer.sep_token_id]
        )
        attn = [1] * len(ids)

        pad_len = self.max_tokens - len(ids)
        if pad_len:
            ids += [self.tokenizer.pad_token_id] * pad_len
            attn += [0] * pad_len

        if self.framework == "pt":
            ids = torch.tensor([ids], device=self.device)
            attn = torch.tensor([attn], device=self.device)
        else:
            ids = tf.constant([ids], dtype=tf.int32)
            attn = tf.constant([attn], dtype=tf.int32)

        return ids, attn


    def encode(self, text: str) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Returns a list of embeddings (one per chunk).  Each embedding is a
        NumPy array with the hidden size of the selected model.
        """
        embeddings = []

        for chunk_ids in self._split_into_chunks(text):
            ids, attn = self._prepare_inputs(chunk_ids)
            outputs = self.model(input_ids=ids, attention_mask=attn)

            if self.framework == "pt":
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.last_hidden_state 

            if self.pooling == "cls":
                emb = hidden[:, 0, :] 
            else:  
                mask = attn.unsqueeze(-1) if self.framework == "pt" else tf.expand_dims(attn, -1)
                summ = (hidden * mask).sum(dim=1) if self.framework == "pt" else tf.reduce_sum(hidden * tf.cast(mask, hidden.dtype), axis=1)
                counts = mask.sum(dim=1) if self.framework == "pt" else tf.reduce_sum(mask, axis=1)
                emb = summ / counts

            emb_np = emb.detach().cpu().numpy() if self.framework == "pt" else emb.numpy()
            embeddings.append(emb_np.squeeze())

        return embeddings
