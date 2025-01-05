"""Define a single custumized model."""

import os
import sys
from pathlib import Path
from abc import ABC
from typing import List
import json

from omegaconf import OmegaConf
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import tools.train_utils as tu
from tools.model_utils import create_model
from tools.metric_utils import compute_der, IterMeter


class BaseModel(ABC):
    """Class that define a custom model.

    Args:
        BASEModel (_type_): _description_
    """

    def __init__(self, cfg: dict, phases: List[str], device: torch.device='cpu') -> None:
        super(BaseModel, self).__init__()

        self.device = device
        self.cfg = cfg
        print(cfg)
        self.model_type = cfg.model_type
        self.model_folder = cfg.model_folder
        self.model_name = cfg.model_name

        self._learning_rate = None
        self._optimizer = None
        self._scheduler = None
        self._criterion = None

        self._phase = None
        self.best_der = float('inf')
        self.best_epoch = 0

        self.train_state = []
        self.train_meter = {}
        for phase in phases:
            self.train_meter[phase] = IterMeter()

        print(f'Create the {self.model_name} model for {self.model_type}')
        self._model, self.epoch, self.best_der, self.learn_rate = create_model(self.cfg)
        self.best_epoch = self.epoch
        self._model.to(self.device)
        
        self.set_optimizer()
        self.set_learning_rate()
        self.set_scheduler()
        self.set_criterion()

    @property
    def parameters(self):
        """Return parameters to be updated."""

        parameters = []
        for param in self._model.parameters():
            if param.requires_grad:
                parameters.append(param)
        return parameters

    @property
    def phase(self):
        """Return the current phase."""

        return self._phase

    @phase.setter
    def phase(self, phase: str=None) -> None:
        """Set the phase."""

        self._phase = phase
        if self._phase == 'train':
            self._model.train()
        else:
            self._model.eval()
        self._reset_metrics()


    @property
    def learning_rate(self):
        """return the learning rate."""

        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set the learning rate."""

        self._learning_rate = value

    def set_learning_rate(self):
        """Set the learning rate"""

        lr = self.cfg.get('learning_rate', None)
        if lr is not None:
            lr_type = 'provided'
        else:
            lr = 0.001
            lr_type = 'default'

        print(f'Set the {lr_type} learning rate: {lr}')
        self.learning_rate= lr

    @property
    def optimizer(self) -> Optimizer:
        """Set the optimizer."""

        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        """Set the optimizer

        Args:
            optimizer (Optimizer): The optimizer
        """

        self._optimizer = value

    def set_optimizer(self):
        """Set the optimizer"""

        if self.cfg.optimizer and self.cfg.optimizer.name is not None:
            opname = self.cfg.optimizer.name
            kwargs = self.cfg.optimizer.get('options', None)
            kwargs.lr = self.learning_rate
            optimizer = tu.set_optimizer(self, opname, **kwargs)
            opt_type = 'provided'
        else:
            opname = 'Adam'
            optimizer = tu.set_optimizer(self, opname)
            opt_type = 'default'

        print(f'Set the {opt_type} optimizer: {optimizer}')

        self.optimizer = optimizer

    @property
    def scheduler(self) -> Optimizer:
        """Return the scheduler."""

        return self._scheduler

    @scheduler.setter
    def scheduler(self, value: Optimizer = 'ReduceLROnPlateau') -> None:
        """Set the scheduler given in the config file or set the default one.

        Args:
            scheduler (Optimizer, optional): _description_. Defaults to 'ReduceLROnPlateau'.
        """

        self._schedeler = value

    def set_scheduler(self) -> None:
        """Set the scheduler."""

        if self.cfg.scheduler and self.cfg.scheduler.name is not None:
            schname = self.cfg.scheduler.name
            kwargs = self.cfg.scheduler.get('options', None)
            scheduler = tu.set_scheduler(self.optimizer, schname, **kwargs)
            sch_type = 'provided'
        else:
            schname = 'ReduceLROnPlateau'
            scheduler = tu.set_scheduler(self.optimizer, schname)
            sch_type = 'default'

        print(f'Set the {sch_type} scheduler {scheduler}')
        self.scheduler = scheduler

    @property
    def criterion(self) -> torch.nn:
        """Return the criterion."""

        return self._criterion

    @criterion.setter
    def criterion(self, value: torch.nn) -> None:
        """Set the training criterion to value.

        Args:
            value (torch.nn, optional): The training criterion.
        """

        self._criterion = value

    def set_criterion(self) -> None:
        """Set the training criterion."""

        if self.cfg.criterion and self.cfg.criterion.name is not None:
            crname = self.cfg.criterion.name
            kwargs = self.cfg.criterion.get('options', None)
            cr_type = 'provided'
            criterion = tu.set_criterion(crname, **kwargs)
        else:
            crname = 'NLLLoss'
            kwargs: {
                'ignore_index': -1
            }
            cr_type = 'default'
            criterion = tu.set_criterion(crname, **kwargs)

        print(f'Set the {cr_type} criterion: {criterion}')
        self.criterion = criterion


    def is_better(self, epoch:int) -> bool:
        """Define metric to compare tow models.

        Args:
            epoch (int): current epoch

        Returns:
            bool: is the current epoch better than the previous best epoch
        """

        _, epoch_der = self.train_meter[self._phase].current_metrics()
        if epoch_der  < self.best_der:
            self.best_der = epoch_der
            self.best_epoch = epoch
            return True
        return False


    def trainer_state_update(self, epoch:int):
        """Update the training status.

        Args:
            epoch (int): The current epoch
        """

        current_loss, current_der = self.train_meter[self._phase].current_metrics()
        self.train_state.append(
            {
                'epoch': epoch,
                f'{self._phase}_loss': f'{current_loss:.3f}',
                f'{self._phase}_der': f'{current_der:.3f}',
                'learning rate': self.optimizer.param_groups[0]['lr']
            }
        )


    def step_update(self, nb_samples: int, loss: float, errors: float) -> None:
        """Update the statistics after each step.

        Args:
            nb_samples (int): number of samples in a batch
            loss (float): the loss value
            errors (float): the edition errors
        """

        self.train_meter[self._phase].update_step_metric(nb_samples, loss, errors)


    def _reset_metrics(self):
        """Reset metrics."""

        self.train_meter[self._phase].reset()


    def run_step(self, batch: torch.Tensor) -> None:
        """Train and validate the model on the datasets.

        Args:
            batch (torch.Tensor): _description_
        """

        inputs, targets, inputs_size, in_diacs  = batch
        inputs = inputs.to(self.device)
        inputs_size = inputs_size.to(self.device)
        targets = targets.to(self.device)
        if in_diacs is not None:
            in_diacs = in_diacs.to(self.device)

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(self.phase == 'train'):
            logits = self._model(inputs, in_diacs)
            logits = logits.permute(0, 2, 1)
            log_probs = F.log_softmax(logits, dim=1)
            loss = self._criterion(log_probs, targets)
            if self._phase == 'train':
                loss.backward()
                self._optimizer.step()

        der, nb_samples = compute_der(log_probs.cpu(), targets.cpu(), inputs_size)
        step_loss = loss.item()
        self.step_update(nb_samples, step_loss, der)


    def write_train_summary(self, train_log_file: str) -> None:
        """Write to a file the training status.

        Args:
            train_log_file (str): Path to the log file
        """

        with open(train_log_file, "w", encoding='utf-8') as fout:
            json.dump(self.train_state, fout, indent=4)


    def save_checkpoint(self, output_folder: str, epoch: int, epoch_loss: float,
                        epoch_der: float) -> None:
        """Save the model parameters.

        Args:
            output_folder (string): Path to save the checkpoint.
            epoch (integer): Epoch number.
            epoch_loss: (float): The epoch loss.
            epoch_der (Counter): Edition Errors rate on the validation dataset.
            if_best_only (bool): save the model only if it is better than the previous one.
        """

        model_cfg = OmegaConf.create(OmegaConf.to_yaml(self.cfg.model_config, resolve=True))
        package = {
                   'epoch': epoch,
                   'cfg': model_cfg,
                   'optimizer_state_dict': self._optimizer.state_dict(),
                   'learning_rate': self._optimizer.param_groups[0]['lr'],
                   'epoch_loss': epoch_loss,
                   'valid_error': epoch_der,
                   'state_dict': self._model.state_dict()
        }
        file_name = f'checkpoint_epoch{epoch}.pt'
        checkpoint = os.path.join(output_folder, self.model_folder, file_name)
        torch.save(package, checkpoint)
