import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import data.dataset as module_dataset
import data.dataloader as module_dataloader
import model as model_module
from model import *
from parse_config import ConfigParser
from tester import *


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('test')
    seed = config.set_seed()
    logger.info('seed: {}'.format(seed))

    test_dataset = config.init_obj('test_dataset', module_dataset, **{'seed': seed})
    data_loader = config.init_obj('data_loader', module_dataloader, **{'dataset': test_dataset})

    model = config.init_obj('arch', model_module)
    logger.debug(model)

    output_dir = Path(config['tester']['save_dir']) / 'test'

    Tester = eval(config['tester']['tester_type'])
    tester = Tester(model,
                    config=config,
                    data_loader=data_loader,
                    output_dir=output_dir)
    tester.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, required=True,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, required=True,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=1234, type=str)

    config = ConfigParser.from_args(args)
    main(config)