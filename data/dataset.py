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


class Dataset(BaseDataset):
    """
    [img & mask] for seg task
    """

    def __init__(self, datasets_root, image_root, mask_root, data_aug_prob, mean, std, seed):
        """Initialize file paths or a list of file names. """
        super(Dataset, self).__init__(seed)
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
                util.RandomCrop(crop_size),
                util.Jitter_HSV(self.data_aug_prob),
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

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class MTLDataset(BaseDataset):
    """
    [img & mask & conn] for multi-task learning
    """
    def __init__(self, datasets_root, image_root, mask_root, conn_root, data_aug_prob, mean, std, seed):
        super(MTLDataset, self).__init__(seed)
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
                util.RandomCrop(crop_size),
                util.Jitter_HSV(self.data_aug_prob),
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

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class HGDataset(BaseDataset):
    """
    [image, mask, 1/4 mask] for Hourglass task.
    """

    def __init__(self, datasets_root, image_root, mask_root, data_aug_prob, mean, std, seed):
        """Initialize file paths or a list of file names. """
        super(HGDataset, self).__init__(seed)
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
                util.RandomCrop(crop_size),
                util.Jitter_HSV(self.data_aug_prob),
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
        w, h, _ = image.shape

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        mask_4 = cv2.resize(mask, (w // 4, h // 4))

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'mask_4': mask_4}

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        _, sample['mask_4'] = cv2.threshold(sample['mask_4'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class MTLHGDataset(BaseDataset):
    """
    [ image, mask, 1/4 mask, conn, 1/4 conn ] for multi-task hourglass task.
    """

    def __init__(self, datasets_root, image_root, mask_root, conn_root, conn_4_root, data_aug_prob, mean, std, seed):
        """Initialize file paths or a list of file names. """
        super(MTLHGDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root
        self.conn_root = self.datasets_root / conn_root
        self.conn_4_root = self.datasets_root / conn_4_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')
        conn_list = util.get_fname_list(self.conn_root, suffix='*.png')
        conn_4_list = util.get_fname_list(self.conn_4_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        conn_names = pd.DataFrame({'conn_fname': conn_list})
        conn_4_names = pd.DataFrame({'conn_4_fname': conn_4_list})

        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]
        conn_names['match_substr'] = [Path(f).stem for f in conn_list]
        conn_4_names['match_substr'] = [Path(f).stem for f in conn_4_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner').merge(
            conn_names, on='match_substr', how='inner').merge(
            conn_4_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomCrop(crop_size),
                util.Jitter_HSV(self.data_aug_prob),
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
        w, h, _ = image.shape

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        mask_4 = cv2.resize(mask, (w // 4, h // 4))
        conn_src = self.match_df.loc[idx, 'conn_fname']
        conn = cv2.imread(conn_src, cv2.IMREAD_GRAYSCALE)
        conn_4_src = self.match_df.loc[idx, 'conn_4_fname']
        conn_4 = cv2.imread(conn_4_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'mask_4': mask_4,
                  'conn': conn,
                  'conn_4': conn_4}

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        _, sample['mask_4'] = cv2.threshold(sample['mask_4'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class ImproveDataset(BaseDataset):
    """
    [ image, mask, 1/4 mask, conn, 1/4 conn ] for multi-task hourglass task.
    """

    def __init__(self, datasets_root, image_root, mask_root, conn_root, conn_2_root, conn_4_root, data_aug_prob, mean, std, seed):
        """Initialize file paths or a list of file names. """
        super(ImproveDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root
        self.conn_root = self.datasets_root / conn_root
        self.conn_2_root = self.datasets_root / conn_2_root
        self.conn_4_root = self.datasets_root / conn_4_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')
        conn_list = util.get_fname_list(self.conn_root, suffix='*.png')
        conn_2_list = util.get_fname_list(self.conn_2_root, suffix='*.png')
        conn_4_list = util.get_fname_list(self.conn_4_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        conn_names = pd.DataFrame({'conn_fname': conn_list})
        conn_2_names = pd.DataFrame({'conn_2_fname': conn_2_list})
        conn_4_names = pd.DataFrame({'conn_4_fname': conn_4_list})

        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]
        conn_names['match_substr'] = [Path(f).stem for f in conn_list]
        conn_2_names['match_substr'] = [Path(f).stem for f in conn_2_list]
        conn_4_names['match_substr'] = [Path(f).stem for f in conn_4_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner').merge(
            conn_names, on='match_substr', how='inner').merge(
            conn_2_names, on='match_substr', how='inner').merge(
            conn_4_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomCrop2(crop_size),
                util.RandomHorizontalFlip(self.data_aug_prob),
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
        w, h, _ = image.shape

        # Corresponding to the given image
        mask_src = self.match_df.loc[idx, 'mask_fname']
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
        mask_2 = cv2.resize(mask, (w // 2, h // 2))
        mask_4 = cv2.resize(mask, (w // 4, h // 4))
        conn_src = self.match_df.loc[idx, 'conn_fname']
        conn = cv2.imread(conn_src, cv2.IMREAD_GRAYSCALE)
        conn_2_src = self.match_df.loc[idx, 'conn_2_fname']
        conn_2 = cv2.imread(conn_2_src, cv2.IMREAD_GRAYSCALE)
        conn_4_src = self.match_df.loc[idx, 'conn_4_fname']
        conn_4 = cv2.imread(conn_4_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'mask_2': mask_2,
                  'mask_4': mask_4,
                  'conn': conn,
                  'conn_2': conn_2,
                  'conn_4': conn_4}

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        _, sample['mask_2'] = cv2.threshold(sample['mask_2'], 127, 1, cv2.THRESH_BINARY)
        _, sample['mask_4'] = cv2.threshold(sample['mask_4'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class MTLDataset3(BaseDataset):
    """
    [image, mask, centerline, point] for 3-task learning.
    """

    def __init__(self, datasets_root, image_root, mask_root, line_root, point_root, data_aug_prob, mean, std, seed):
        super(MTLDataset3, self).__init__(seed)
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
                util.RandomCrop(crop_size),
                util.Jitter_HSV(self.data_aug_prob),
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

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        sample['line'] = sample['line'] / 255.
        sample['point'] = sample['point'] / 255

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


class XGDataset(BaseDataset):
    """
    [image, mask, centerline, point] for 3-task learning.
    """

    def __init__(self, datasets_root, image_root, mask_root, edge_root, mini_root, direct_root, data_aug_prob, mean, std, seed):
        super(XGDataset, self).__init__(seed)
        self.datasets_root = Path(datasets_root)
        self.image_root = self.datasets_root / image_root
        self.mask_root = self.datasets_root / mask_root
        self.edge_root = self.datasets_root / edge_root
        self.mini_root = self.datasets_root / mini_root
        self.direct_root = self.datasets_root / direct_root

        # Preparing images and labels filename lists.
        image_list = util.get_fname_list(self.image_root)
        mask_list = util.get_fname_list(self.mask_root, suffix='*.png')
        edge_list = util.get_fname_list(self.edge_root, suffix='*.png')
        mini_list = util.get_fname_list(self.mini_root, suffix='*.png')
        direct_list = util.get_fname_list(self.direct_root, suffix='*.png')

        image_names = pd.DataFrame({'image_fname': image_list})
        mask_names = pd.DataFrame({'mask_fname': mask_list})
        edge_names = pd.DataFrame({'edge_fname': edge_list})
        mini_names = pd.DataFrame({'mini_fname': mini_list})
        direct_names = pd.DataFrame({'direct_fname': direct_list})

        image_names['match_substr'] = [Path(f).stem for f in image_list]
        mask_names['match_substr'] = [Path(f).stem for f in mask_list]
        edge_names['match_substr'] = [Path(f).stem for f in edge_list]
        mini_names['match_substr'] = [Path(f).stem for f in mini_list]
        direct_names['match_substr'] = [Path(f).stem for f in direct_list]

        self.match_df = image_names.merge(mask_names, on='match_substr', how='inner').merge(
            edge_names, on='match_substr', how='inner').merge(
            mini_names, on='match_substr', how='inner').merge(
            direct_names, on='match_substr', how='inner')

        # mean & std
        self.mean = eval(mean)
        self.std = eval(std)
        self.normalize = transforms.Normalize(self.mean, self.std)

        self.data_aug_prob = data_aug_prob
        self.transform = None
        if self.data_aug_prob > 0:
            self.transform = transforms.Compose([
                util.RandomCrop2(crop_size),
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
        edge_src = self.match_df.loc[idx, 'edge_fname']
        edge = cv2.imread(edge_src, cv2.IMREAD_GRAYSCALE)
        mini_src = self.match_df.loc[idx, 'mini_fname']
        mini = cv2.imread(mini_src, cv2.IMREAD_GRAYSCALE)
        direct_src = self.match_df.loc[idx, 'direct_fname']
        direct = cv2.imread(direct_src, cv2.IMREAD_GRAYSCALE)

        sample = {'image_name': image_name,
                  'image': image,
                  'mask': mask,
                  'edge': edge,
                  'mini': mini,
                  'direct': direct}

        # Online Data Augmentation
        if self.transform:
            sample = self.transform(sample)

        _, sample['mask'] = cv2.threshold(sample['mask'], 127, 1, cv2.THRESH_BINARY)
        _, sample['edge'] = cv2.threshold(sample['edge'], 127, 1, cv2.THRESH_BINARY)
        _, sample['mini'] = cv2.threshold(sample['mini'], 127, 1, cv2.THRESH_BINARY)

        # totensor
        totensor = util.ToTensor()
        sample = totensor(sample)

        sample['image'] = self.normalize(sample['image'])

        return sample


if __name__ == '__main__':
    dataset = XGDataset(
        datasets_root="/home/data/xyj/spacenet/valid",
        image_root="images",
        mask_root="masks",
        edge_root="edge",
        mini_root="mini",
        direct_root="direct",
        data_aug_prob=0,
        mean="[0.334, 0.329, 0.326]",
        std="[0.161, 0.153, 0.144]",
        seed=1234
    )
    dataset[0]

    # plot sample
    # fig, ax = plt.subplots(1, len(sample) - 1)
    # for i, im in enumerate(sample.values()):
    #     if isinstance(im, str):
    #         continue
    #     elif im.shape[0] == 3:
    #         ax[i - 1].imshow(im.numpy().transpose([1, 2, 0]))
    #     else:
    #         ax[i - 1].imshow(im.numpy()[0])
