"""classes and method to help training the recognizer.
https://github.com/TorchEnsemble-Community/Ensemble-Pytorch/blob/master/torchensemble/utils/set_module.py
"""

import importlib
from torch.optim import Optimizer
import torch.nn as nn


def set_optimizer(model: nn.Module, optimizer_name: Optimizer, **kwargs) -> Optimizer:
    """Set the parameters of the optimizer for the model.

    Args:
        model (nn.Module): The model
        optimizer_name (Optimizer): The optimizer name

    Returns:
        Optimizer: The optimizer

    Reference:
        https://pytorch.org/docs/stable/optim.html#algorithms
    """

    supported_optimizers = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "RMSprop",
        "SGD",
    ]

    if optimizer_name not in supported_optimizers:
        raise NotImplementedError(
            f'Unrecognized optimizer: {optimizer_name}, \
                should be one of {",".join(supported_optimizers)}.'
        )

    optimizer_cls = getattr(
        importlib.import_module("torch.optim"), optimizer_name
    )
    optimizer = optimizer_cls(model.parameters, **kwargs)
    return optimizer


def set_scheduler(optimizer: Optimizer, scheduler_name: Optimizer, **kwargs) -> Optimizer:
    """Set the scheduler on learning rate for the optimizer.

    Args:
        optimizer (Optimizer): The optimizer
        scheduler_name (Optimizer): The scheduler name

    Raises:
        NotImplementedError: _description_

    Returns:
        Optimizer: The scheduler

    Reference:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    """

    supported_lr_schedulers = [
        "LambdaLR",
        "MultiplicativeLR",
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
        "CyclicLR",
        "OneCycleLR",
        "CosineAnnealingWarmRestarts",
    ]

    if scheduler_name not in supported_lr_schedulers:
        raise NotImplementedError(
            f'Unrecognized scheduler: {scheduler_name}, \
                should be one of {",".join(supported_lr_schedulers)}.'
        )

    scheduler_cls = getattr(
        importlib.import_module("torch.optim.lr_scheduler"), scheduler_name
    )
    scheduler = scheduler_cls(optimizer, **kwargs)
    return scheduler


def set_criterion(criterion_name: nn, **kwargs) -> nn:
    """Set the training criterion.

    Args:
        criterion_name (nn): Name of the criterion

    Returns:
        nn: The criterion
    """

    supported_criteria = [
        'NLLLoss',
        'CrossEntropyLoss'
    ]

    if criterion_name not in supported_criteria:
        raise NotImplementedError(
            f'Unrecognized criterion: {criterion_name}, \
                should be one of {",".join(supported_criteria)}.'
        )
    criterion_cls = getattr(
        importlib.import_module('torch.nn'), criterion_name
    )
    criterion = criterion_cls(**kwargs)
    return criterion
