"""Define RNN-based model for diacritic restoration."""

import torch
import torch.nn as nn

supported_rnns = {'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU, 'rnn': torch.nn.RNN}

class RNN(nn.Module):
    """Define the RNN-based model."""

    def __init__(self, rnn_config: dict):
        super(RNN, self).__init__()

        self.rnn_type = supported_rnns[rnn_config.rnn_type]
        self.rnns = self.rnn_type(
            input_size = rnn_config.input_size,
            hidden_size = rnn_config.hidden_size,
            num_layers = rnn_config.n_layers,
            bidirectional = rnn_config.bidirectional,
            batch_first = True,
            dropout= rnn_config.dropout,
            bias = False
        )

        num_directions = 2 if rnn_config.bidirectional else 1
        classif_input_size = num_directions * rnn_config.hidden_size
        self.classifier = torch.nn.Linear(classif_input_size, rnn_config.n_classes)
        self.apply(init_rnn)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the outputs for a given input sample.

        Args:
            input_tokens (torch.Tensor): Token indices

        Returns:
            torch.Tensor: Logits for diacritics classes.
        """

        outputs, _ = self.rnns(inputs)
        logits = self.classifier(outputs)
        return logits


def init_rnn(module):
    """Initialize weights for Seq2Seq."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    if isinstance(module, nn.GRU):
        for param in module._flat_weights_names:
            if 'weight' in param:
                nn.init.xavier_uniform_(module._parameters[param])
