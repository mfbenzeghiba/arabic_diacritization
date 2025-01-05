"""Script that train the diacritizer model."""

import os
import sys
from pathlib import Path
import argparse
import shutil
import time
import logging
import json

from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from data_processing.load_data import Tokenizer, Diacritizer, Tashkeel
from models.base_model import BaseModel


def main(cfg_path: str, options: dict):
    """Train the diacritizer

    Args:
        options (dict): options to train the diacritizer
    """

    common_options = options.common
    data_options = options.datasets
    exp_folder = common_options.exp_folder

    seed = common_options.seed
    torch.manual_seed(seed)

    shutil.copy(cfg_path, exp_folder)

    use_cuda = common_options.use_cuda
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(data_options.char_vocab)
    vocab_size = len(tokenizer)

    diacritizer = Diacritizer(data_options.diac_vocab)
    target_size = len(diacritizer)

    train_dataset = Tashkeel(data_options.train, tokenizer, diacritizer, data_options.partial_prob)
    valid_dataset = Tashkeel(data_options.valid, tokenizer, diacritizer)

    kwargs = {}
    if use_cuda:
        kwargs = {'num_workers': common_options.num_workers, 'pin_memory': True}

    train_loader = DataLoader(train_dataset,
                              batch_size=data_options.batch_size,
                              collate_fn=train_dataset.fn_collate,
                              shuffle=data_options.shuffle_train,
                              **kwargs)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=data_options.batch_size,
                              collate_fn=valid_dataset.fn_collate,
                              shuffle=False,
                              **kwargs)

    model_options = options.model
    model_config = model_options.model_config
    model_config.vocab_size = vocab_size
    model_config.n_classes = target_size
    model = BaseModel(cfg=model_options, phases=common_options.phases, device=device)

    total_params = sum( param.numel() for param in model.parameters )
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.info("-" * 50)
    logging.info('Train data: %s', data_options.train)
    logging.info('Valid data: %s', data_options.valid)
    logging.info('Number of training batch: %d', len(train_loader))
    logging.info('Number of validation batch: %d', len(valid_loader))
    logging.info('vocabulary size: %d', vocab_size)
    logging.info('Number of classes: %d', target_size)
    logging.info('Experiment folder: %s', exp_folder)
    logging.info('PyTorch Version: %s', torch.__version__)
    logging.info('Model type: %s', model_options.model_type)
    logging.info('Model name: %s', model_options.model_name)
    logging.info('Number of parameters: %d', total_params )
    logging.info('Device: %s', device)
    logging.info("-" * 50)

    model_folder = model_options.model_folder
    model_folder = os.path.join(exp_folder, model_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        raise EOFError(f'{model_folder} already exists')

    logging.info('Model folder: %s', model_folder)
    phases = common_options.phases
    bepoch = model.epoch # in case of continous training
    nepochs = common_options.nepochs
    begin_train_time = time.time()

    training_log_file = os.path.join(exp_folder, 'training_status.json')
    for epoch in range(bepoch+1, nepochs):
        logging.info('Epoch %d', epoch)
        for _, phase in enumerate(phases):
            logging.info('%s phase....', phase)
            model.phase = phase
            eval_meter = model.train_meter[model.phase]
            if phase == 'train':
                for batch in tqdm(train_loader):
                    model.run_step(batch)
                epoch_train_loss, epoch_train_errors = eval_meter.current_metrics()
                logging.info('Model: %s\t loss: %.3f\t token error: %.3f',
                             model.model_name, epoch_train_loss, epoch_train_errors)
            else:
                for batch in tqdm(valid_loader):
                    model.run_step(batch)
                epoch_valid_loss, epoch_valid_errors = eval_meter.current_metrics()
                logging.info('Model: %s\t loss: %.3f\t token error: %.3f',
                             model.model_name, epoch_valid_loss, epoch_valid_errors
                )
                if model.is_better(epoch):
                    model.save_checkpoint(
                        exp_folder, epoch, epoch_valid_loss, epoch_valid_errors
                    )
                if model.scheduler is not None:
                    model.scheduler.step(epoch_valid_loss)
            model.trainer_state_update(epoch)
        model.write_train_summary(training_log_file)

    end_train_time = time.time()
    train_time = (end_train_time - begin_train_time) / 3600
    train_summary = {
        'model_name': model.model_name,
        'best_checkpoint': os.path.join(model_folder, f'checkpoint_epoch{model.best_epoch}.pt'),
        'best_epoch': model.best_epoch,
        'best_error': f'{model.best_der:.3f}',
        'learning_rate': model.optimizer.param_groups[0]['lr'],
        'batch_size': data_options.batch_size,
        'nb_batches': len(train_loader),
        'training_time': f'{train_time:.3f}'
    }

    model.train_state.insert(0, train_summary)
    with open(training_log_file, "w", encoding='utf-8') as fout:
        json.dump(model.train_state, fout, indent=4)
    print(model.train_state)

    logging.info(f'Training lasts {train_time // 60:.0f}m {train_time%60 :.0f}s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script to train attention models.')
    parser.add_argument('--config', default='conf/lstm_config.yaml',
                        help='conf file with argument of LSTM and training')
    args = parser.parse_args()

    config_path = args.config
    try:
        opts = OmegaConf.load(config_path)
    except Exception as e:
        logging.error(e)

    exp_dir = opts.common.exp_folder
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    main(config_path, opts)
