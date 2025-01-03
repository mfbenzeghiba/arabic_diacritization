"""https://github.com/Diamondfan/CTC_pytorch/blob/master/timit/steps/train_ctc.py."""

from typing import Tuple

import torch
from models.tashkeel_rnn import TashkeelRNN


def define_model(model_cfg: dict, model_type: str):
    """Define the architecture of the model from the config dict.

    Args:
        cfg (dict): _description_
    """
    if model_type == 'rnn':
        return TashkeelRNN(model_cfg)

def create_model(cfg: dict) -> Tuple[torch.nn.Module, int, float, float]:
    """Implement the model.

    Args:
        model_name (str): The name of the model.
        cfg (dict): Dictionary contains the model options.

    Returns:
        torch.nn.Module: the model
        bepoch (int): The begining epoch
        best_error (float): The error of the loaded checkpoint
        learn_rate (float): The learning rate of the checkpoint.
    """

    bepoch = 0
    best_error = float('inf')
    learn_rate = None

    model_type = cfg.model_type.lower()
    model = define_model(cfg.model_config, model_type)
    model_path = cfg.from_pretrained
    if model_path is not None:
        tl = torch.load(model_path, map_location='cpu')
        bepoch = tl.get('epoch', 0)
        best_error = tl['valid_error']['ter']
        learn_rate = tl['learning_rate']
        model.load_state_dict(tl['state_dict'])
    return model, bepoch, best_error, learn_rate


def load_checkpoint(model_path: str, model_type: str):
    """Load a previously trained model.

    Args:
        model_path (str, optional): The path to the model.
    """

    checkpoint = torch.load(model_path, map_location='cpu')
    cfg = checkpoint['cfg']
    model = define_model(cfg, model_type)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    return model
