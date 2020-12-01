import cv2
import torch
import numpy as np
from pathlib import Path


class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        for k in sample.keys():
            if k == 'image_name':
                continue
            sample[k] = cv2.resize(sample[k], (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        return sample


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        new_h, new_w = self.output_size
        new_h_4, new_w_4 = new_h // 4, new_w // 4

        for i in range(10):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            if np.any(sample['image'][top: top + new_h, left: left + new_w]):
                break

        if i == 9:
            max = 0
            for i, j in [(x, y) for x in [0, 394, 788] for y in [0, 394, 788]]:
                if (sample['image'][i:i+512, j:j+512]).sum() > max:
                    max = (sample['image'][i:i+512, j:j+512]).sum()
                    top = i
                    left = j

        top_4 = top // 4
        left_4 = left // 4

        for k in sample.keys():
            if k == 'image_name':
                continue
            elif k == 'mask_4' or k == 'line_4':
                sample[k] = sample[k][top_4: top_4 + new_h_4,
                left_4: left_4 + new_w_4]
            else:
                sample[k] = sample[k][top: top + new_h,
                            left: left + new_w]

        return sample


class RandomHorizontalFlip:
    def __init__(self, u=0):
        self.u = u

    def __call__(self, sample):
        if np.random.random() < self.u:
            for k in sample.keys():
                if k == 'image_name':
                    continue
                sample[k] = np.fliplr(sample[k])
        return sample


class RandomVerticleFlip:
    def __init__(self, u=0):
        self.u = u

    def __call__(self, sample):
        if np.random.random() < self.u:
            for k in sample.keys():
                if k == 'image_name':
                    continue
                sample[k] = np.flipud(sample[k])
        return sample


class RandomRotate90:
    """Anti-clockwise roation with 90 * k (k = 1, 2, 3)"""
    def __init__(self, u=0):
        self.u = u

    def __call__(self, sample):
        if np.random.random() < self.u:
            k_ = np.random.randint(1, 4)
            for k in sample.keys():
                if k == 'image_name':
                    continue
                sample[k] = np.rot90(sample[k], k_)
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for k in sample.keys():
            if k == 'image_name':
                continue
            elif k == 'image':
                sample[k] = torch.from_numpy((sample[k]/255.0).transpose((2, 0, 1)).astype(np.float32))
            else:
                sample[k] = torch.from_numpy(sample[k][np.newaxis, :].astype(np.float32))

        return sample


def get_fname_list(p, suffix='*.tif'):
    """Get a list of filenames from p, which can be a dir, fname, or list."""
    if isinstance(p, list):
        return p
    elif isinstance(p, Path):
        if p.is_dir():
            return [str(f) for f in p.glob(suffix)]
        elif p.is_file():
            return [str(p)]
        else:
            raise ValueError(f"If a `pathlib.Path` is provided, it must be a valid"
                             f" path.{p} is invalid.")
    else:
        raise ValueError("{} is not a string or list.".format(p))
