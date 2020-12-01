import torch
import argparse

import data.dataset as dataset_module
import data.dataloader as dataloader_module
import model as model_module
from model import *
from trainer import *
from parse_config import ConfigParser
from utils import weights_init


def main(config):
    logger = config.get_logger(name='train')
    # fix random seeds for reproducibility
    seed = config.set_seed()
    logger.info('seed: {}'.format(seed))

    train_dataset = config.init_obj('train_dataset', dataset_module, **{'seed': seed})
    valid_dataset = config.init_obj('valid_dataset', dataset_module, **{'seed': seed})

    data_loader = config.init_obj('data_loader', dataloader_module, **{'dataset': train_dataset})
    valid_data_loader = config.init_obj('data_loader', dataloader_module, **{'dataset': valid_dataset})

    model = config.init_obj('arch', model_module)
    weights_init(model, seed=seed)
    logger.debug(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    Trainer = eval(config['trainer']['trainer_type'])
    trainer = Trainer(model, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='path to config file (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=1234, type=str)

    config = ConfigParser.from_args(args)
    main(config)
