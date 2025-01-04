"""Define several classes to concatenate embeddings."""

import torch
import torch.nn as nn

class BaseEmbedding(nn.Module):
    """Define the embedding module."""
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super(BaseEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)

    def forward(self, tokens_inputs: torch.Tensor) -> torch.Tensor:
        """Compute the embeddings for an input tensor.

        Args:
            inputs (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The embeddings
        """

        return self.embedding(tokens_inputs)


class SumEmbedding(nn.Module):
    """Sum embeddings of sevral inputs."""

    def __init__(self, token_vocab_size: int, diac_vocab_size: int, embedding_size: int):
        super(SumEmbedding, self).__init__()
        self.token_embedding = BaseEmbedding(token_vocab_size, embedding_size)
        self.diac_embedding = BaseEmbedding(diac_vocab_size, embedding_size)
        self.embedding_size = 2 * embedding_size

    def forward(self, token_inputs: torch.Tensor, diac_inputs: torch.Tensor) -> torch.Tensor:
        """Compute the embeddings for input tensors.

        Args:
            token_inputs (torch.Tensor): The input tensor for tokens
            diac_inputs (torch.Tensor): The input tensor for diacritics

        Returns:
            torch.Tensor: sum of the tokens and diac embeddings
        """

        token_embeddings = self.token_embedding(token_inputs)
        diac_embeddings = self.diac_embedding(diac_inputs)
    
        return token_embeddings + diac_embeddings


class ConcatinateEmbedding(nn.Module):
    """Concatinate embeddings."""

    def __init__(self, token_vocab_size: int, diac_vocab_size:int, embedding_size: int):
        super(ConcatinateEmbedding, self).__init__()
        self.token_embedding = BaseEmbedding(token_vocab_size, embedding_size)
        self.diac_embedding = BaseEmbedding(diac_vocab_size, embedding_size)

    def forward(self, token_inputs: torch.Tensor, diac_inputs: torch.Tensor) -> torch.Tensor:
        """Compute the embeddings for input tensors.

        Args:
            token_inputs (torch.Tensor): The input tensor for tokens
            diac_inputs (torch.Tensor): The input tensor for diacritics

        Returns:
            torch.Tensor: Copncatenation of the token and diac embeddings
        """
        token_embeddings = self.token_embedding(token_inputs)
        diac_embeddings = self.diac_embedding(diac_inputs)
        
        return torch.cat((token_embeddings, diac_embeddings), dim=-1)
