import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

import data.datautils as util
from base import BaseDataset

crop_size = 256


class SDataset(BaseDataset):
    """
    Small dataset (original images crop into 256), img & mask for seg task
    """
    def __init__(self, datasets_root, image_root, mask_root, data_aug_prob, mean, std, seed):
        """Initialize file paths or a list of file names. """
        super(SDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomHorizontalFlip(self.data_aug_prob),
                util.RandomVerticleFlip(self.data_aug_prob),
                util.RandomRotate90(self.data_aug_prob)
            ])

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.match_df)

    def __getitem__(self, idx):
        """Return a data pair."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loading a image given a random integer index
        image_src = self.match_df.loc[idx, 'image_fname']
        image_name = self.match_df.loc[idx, 'match_substr']
        image = cv2.imread(image_src, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask}

        rescale = util.RandomCrop(crop_size)
        sample = rescale(sample)

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        # sample['image'] = sample['image'] - np.array(self.mean)*255

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class SMTLDataset(BaseDataset):
    def __init__(self, datasets_root, image_root, mask_root, conn_root, data_aug_prob, mean, std, seed):
        super(SMTLDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root
        self.conn_root = self.datasets_root / conn_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')
        conn_list = util.get_fname_list(self.conn_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        conn_names = pd.DataFrame({'conn_fname': conn_list})

        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]
        conn_names['match_substr'] = [Path(f).stem for f in conn_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner').merge(
            conn_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomHorizontalFlip(self.data_aug_prob),
                util.RandomVerticleFlip(self.data_aug_prob),
                util.RandomRotate90(self.data_aug_prob)
            ])

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.match_df)

    def __getitem__(self, idx):
        """Return a data pair."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loading a image given a random integer index
        image_src = self.match_df.loc[idx, 'image_fname']
        image_name = self.match_df.loc[idx, 'match_substr']
        image = cv2.imread(image_src, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        conn_src = self.match_df.loc[idx, 'conn_fname']
        conn = cv2.imread(conn_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'conn': conn}

        rescale = util.RandomCrop(crop_size)
        sample = rescale(sample)

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class SMTLDataset3(BaseDataset):
    def __init__(self, datasets_root, image_root, mask_root, line_root, point_root, data_aug_prob, mean, std, seed):
        super(SMTLDataset3, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root
        self.line_root = self.datasets_root / line_root
        self.point_root = self.datasets_root / point_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')
        line_list = util.get_fname_list(self.line_root, suffix='*.png')
        point_list = util.get_fname_list(self.point_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        line_names = pd.DataFrame({'line_fname': line_list})
        point_names = pd.DataFrame({'point_fname': point_list})

        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]
        line_names['match_substr'] = [Path(f).stem for f in line_list]
        point_names['match_substr'] = [Path(f).stem for f in point_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner').merge(
            line_names, on='match_substr', how='inner').merge(
            point_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomHorizontalFlip(self.data_aug_prob),
                util.RandomVerticleFlip(self.data_aug_prob),
                util.RandomRotate90(self.data_aug_prob)
            ])

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.match_df)

    def __getitem__(self, idx):
        """Return a data pair."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Loading a image given a random integer index
        image_src = self.match_df.loc[idx, 'image_fname']
        image_name = self.match_df.loc[idx, 'match_substr']
        image = cv2.imread(image_src, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        line_src = self.match_df.loc[idx, 'line_fname']
        line = cv2.imread(line_src, cv2.IMREAD_GRAYSCALE)
        point_src = self.match_df.loc[idx, 'point_fname']
        point = cv2.imread(point_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'line': line,
                  'point': point}

        rescale = util.RandomCrop(crop_size)
        sample = rescale(sample)

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)

        # sample['image'] = self.normalize(sample['image'])
        sample['image'] = sample['image'] - np.array(self.mean) * 255
        sample['line'] = sample['line'] / 255.
        sample['point'] = sample['point'] / 255

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample