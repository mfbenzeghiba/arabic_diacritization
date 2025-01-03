"""Classes and methods to measure the performance of the models."""

from typing import List, Tuple
import torch

class IterMeter(object):
    """keeps track of total iterations"""

    def __init__(self):

        self.nb_steps = 0
        self.nb_samples = 0
        self.loss = 0.
        self.der = 0.

    def update_step_metric(self,
                           nb_samples: int,
                           step_loss: float,
                           step_der: float = None) -> None:
        """Update the evaluation metrics 

        Args:
            nb_samples (int): Number of characters in the batch
            step_loss (float): The loss value
            step_der (float, optional): The DER value. Defaults to None.
        """

        self.nb_samples += nb_samples
        self.nb_steps += 1
        self.loss += step_loss
        if step_der is not None:
            self.der += step_der

    def reset(self) -> None:
        """Reset the metrics."""

        self.der = 0.
        self.loss = 0.
        self.nb_steps = 0
        self.nb_samples = 0

    def current_metrics(self) -> Tuple[float, float]:
        """Return the current metrics."""

        return self.loss/self.nb_steps, self.der/self.nb_samples


def compute_der(preds: torch.Tensor, refs: torch.Tensor,
                inputs_size: torch.Tensor) -> Tuple[float, int]:
    """Compute the diacritic error rate.

    Args:
        hyps (torch.Tensor): The hypotheses text
        refs (torch.Tensor): The reference text
        input_size (torch.Tensor): atensor vector with the length of the input sequences

    """

    der_eval = []
    arg_maxes = torch.argmax(preds, dim=1)
    for i, _ in enumerate(arg_maxes):
        ref = refs[i][:inputs_size[i]]
        pred = arg_maxes[i][:inputs_size[i]]
        der_eval.extend(torch.ne(ref, pred))
    der = sum(der_eval)
    nb_samples = len(der_eval)
    return der, nb_samples


def merge_text_and_diacs(text: List[str], diacs: List[str]) -> str:
    """Merge non diacritized text with diacritics.

    Args:
        text (List): Non diacritized text
        diacs (List): list of diacritics.

    Returns:
        str: Text with diacritics.
    """

    output = []
    for char, diac in zip(text, diacs):
        if diac == '~':
            diac = ''
        if char == '~':
            char = ' '
        output.append(char + diac)
    return ''.join(output)
