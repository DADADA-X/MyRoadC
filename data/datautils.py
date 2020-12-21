import cv2
import torch
import numpy as np
from pathlib import Path


class Jitter_HSV:
    def __init__(self, u=0):
        self.u = u

    def get_params(self, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15)):
        return {'hue_shift': np.random.uniform(hue_shift_limit[0], hue_shift_limit[1]),
                'sat_shift': np.random.uniform(sat_shift_limit[0], sat_shift_limit[1]),
                'val_shift': np.random.uniform(val_shift_limit[0], val_shift_limit[1])}

    def fix_shift_values(self, img, *args):
        """
        shift values are normally specified in uint, but if your data is float - you need to remap values
        """
        if np.ceil(img.max()) == 1:
            return list(map(lambda x: x / 255, args))
        return args

    def shift_hsv(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        maxval = np.max(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
        h, s, v = cv2.split(img)
        h = cv2.add(h, hue_shift)
        h = np.where(h < 0, maxval - h, h)
        h = np.where(h > maxval, h - maxval, h)
        h = h.astype(dtype)
        s = clip(cv2.add(s, sat_shift), dtype, maxval)
        v = clip(cv2.add(v, val_shift), dtype, maxval)
        img = cv2.merge((h, s, v)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, sample):
        if np.random.random() < self.u:
            params = self.get_params()
            for k in sample.keys():
                if k == 'image':
                    hue_shift, sat_shift, val_shift = self.fix_shift_values(sample[k], *params.values())
                    sample[k] = self.shift_hsv(sample[k], hue_shift, sat_shift, val_shift)
                else:
                    continue
        return sample


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


class RandomCrop2:
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
        new_h_2, new_w_2 = new_h // 2, new_w // 2
        new_h_4, new_w_4 = new_h // 4, new_w // 4
        new_h_16, new_w_16 = new_h // 16, new_w // 16

        i, top, left = 0, 0, 0
        for i in range(10):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            if np.any(sample['image'][top: top + new_h, left: left + new_w]):
                break

        if i == 9:
            max = 0
            for t, l in [(x, y) for x in [0, 256] for y in [0, 256]]:
                if (sample['image'][t:t+256, l:l+256]).sum() > max:
                    max = (sample['image'][t:t+256, l:l+256]).sum()
                    top, left = t, l
        top_2 = top // 2
        left_2 = left // 2
        top_4 = top // 4
        left_4 = left // 4
        top_16 = top // 164
        left_16 = left // 16

        for k in sample.keys():
            if k == 'image_name':
                continue
            elif '_4' in k:
                sample[k] = sample[k][top_4: top_4 + new_h_4,
                left_4: left_4 + new_w_4]
            elif '_2' in k:
                sample[k] = sample[k][top_2: top_2 + new_h_2,
                left_2: left_2 + new_w_2]
            elif 'mini' in k:
                sample[k] = sample[k][top_16: top_16 + new_h_16,
                left_16: left_16 + new_w_16]
            else:
                sample[k] = sample[k][top: top + new_h,
                            left: left + new_w]

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

        i, top, left = 0, 0, 0
        for i in range(10):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            if np.any(sample['image'][top: top + new_h, left: left + new_w]):
                break

        if i == 9:
            max = 0
            for t, l in [(x, y) for x in [0, 256] for y in [0, 256]]:
                if (sample['image'][t:t+256, l:l+256]).sum() > max:
                    max = (sample['image'][t:t+256, l:l+256]).sum()
                    top, left = t, l

        top_4 = top // 4
        left_4 = left // 4

        for k in sample.keys():
            if k == 'image_name':
                continue
            elif '_4' in k:
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


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)