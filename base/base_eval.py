import torch
from abc import abstractmethod
from numpy import inf


class BaseEval:
    def __init__(self, model, config):
        self.config = config
        self.logger = config.get_logger('tester')

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        cfg_tester = config['tester']
        self.save_dir = config.save_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if len(self.device_ids) == 1 and k.startswith('module.'):
                state_dict[k[len("module."):]] = state_dict[k]
                del state_dict[k]
            elif len(self.device_ids) > 1 and (not k.startswith('module.')):
                state_dict['module.' + k] = state_dict[k]
                del state_dict[k]

        self.model.load_state_dict(state_dict)

        self.logger.info("Checkpoint loaded. ")
