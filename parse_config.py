import os
import torch
import logging
import numpy as np
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, seed=None, save_dir=None):
        self._config = config
        self.resume = resume
        self.seed = seed

        if 'trainer' in self.config.keys():
            save_dir = Path(self.config['trainer']['save_dir'])
        elif 'tester' in self.config.keys():
            save_dir = Path(self.config['tester']['save_dir'])

        self._save_dir = save_dir

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args):
        """
        Initialize this class from some cli arguments.
        """
        if not isinstance(args, tuple):
            args = args.parse_args()

        config = read_json(args.config)
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            resume = str(resume)
        else:
            resume = None
        seed = args.seed

        return cls(config, resume, seed)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        if name == 'lr_scheduler' and 'milestones' in module_args:
            module_args['milestones'] = eval(module_args['milestones'])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name=None):
        if name == 'train' or name == 'test':
            self._save_dir = self._save_dir / name
            self.save_dir.mkdir(parents=True, exist_ok=True)
        verbosity = self.config['verbosity']
        setup_logging(self.save_dir)
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def set_seed(self):
        if self.seed is None:
            self.seed = np.random.randint(1, 10000)
        else:
            self.seed = int(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        return self.seed

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir
