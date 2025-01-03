"""Define the RNN based tashkeel model."""

import torch
import torch.nn as nn
from modules.rnn import RNN
import modules.composed_embeddings as ce


class TashkeelRNN(nn.Module):
    """Define the LSTM model."""

    def __init__(self, model_config: dict):
        super(TashkeelRNN, self).__init__()
        self.embedding_size = model_config.embedding_size
        self.embedding_name = model_config.embedding_name
        self.vocab_size = model_config.vocab_size

        if self.embedding_name is None:
            # Embedding only tokens (characters)
            self.embedding = ce.BaseEmbedding(self.vocab_size, self.embedding_size)
        else:
            # Embedding both characters and diacritics
            n_classes = model_config.n_classes
            embedding_class = getattr(ce, self.embedding_name)
            self.embedding = embedding_class(self.vocab_size, n_classes, self.embedding_size)

        self.rnns = RNN(model_config)

    def forward(self, tokens_tensor: torch.Tensor,
                diac_tensor: torch.Tensor = None) -> torch.Tensor:
        """Compute the outputs for a given input sample.

        Args:
            tokens_tensor (torch.Tensor): Token indices
            diac_tensor (torch.Tensor): Diacritic indices (optional)

        Returns:
            torch.Tensor: Logits for diacritics classes.
        """

        if self.embedding_name is None:
            embeddings = self.embedding(tokens_tensor)
        else:
            embeddings = self.embedding(tokens_tensor, diac_tensor)
        logits = self.rnns(embeddings)
        return logits
