from __future__ import division
import torch
import random
import cv2
import numpy as np
import numbers
import collections
import math
from .color_jitter import ColorJitterImg


class Compose(object):
    """
    Composes several img_transforms together.

    Args:
        img_transforms (List[Transform]): list of img_transforms to compose.

    Example:
        img_transforms.Compose([
            img_transforms.CenterCrop(10),
            img_transforms.ToTensor(),
        ])
    """

    def __init__(self, img_transforms):
        self.img_transforms = img_transforms

    def __call__(self, img):
        for t in self.img_transforms:
            img = t(img)
        return img


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to torch.FloatTensor of shape (C x H x W)

    def __call__(self, pic):
        if not isinstance(pic, np.ndarray):
            raise (RuntimeError("img_transforms.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(pic.shape) > 3 or len(pic.shape) < 2:
            raise (RuntimeError("img_transforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(pic.shape) == 2:
            pic = np.expand_dims(pic, axis=2)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        return img


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, img):
        assert img.size(0) == len(self.mean)
        if self.std is None:
            for t, m in zip(img, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)
        return img


class Resize(object):
    """
    Resize the input ndarray image with H*W*C or H*W to the given 'size'.
    'size' is a 2-element tuple or list in the order of (width, height)
    """

    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.W = size[0]
        self.H = size[1]

    def __call__(self, img):
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return img


class Crop(object):
    """
    Crops the given ndarray image (H*W*C or H*W) to have a region of the given 'size'.
    'size' can be a tuple (target_height, target_width), a list [target_height, target_width]
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, crop_type='center', padding=None):
        if isinstance(size, int):
            self.crop_W = size
            self.crop_H = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_W = size[0]
            self.crop_H = size[1]
        else:
            raise (RuntimeError("imgseg_transforms.CenterCrop() size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))

    def __call__(self, img):
        img_height, img_width = img.shape[0:2]
        img_channel = 1
        if len(img.shape) == 3:
            img_channel = img.shape[2]
        pad_height = max(self.crop_H - img_height, 0)
        pad_width = max(self.crop_W - img_width, 0)
        pad_h_half = int(pad_height / 2)
        pad_w_half = int(pad_width / 2)
        if pad_height > 0 or pad_width > 0:
            if self.padding is None:
                raise (RuntimeError("imgseg_transforms.Crop() need padding while padding argument is None\n"))
            if img_channel != len(self.padding):
                raise (RuntimeError("padding channel is not equal with image channel\n"))
            img = cv2.copyMakeBorder(img, pad_h_half, pad_height - pad_h_half, pad_w_half, pad_width - pad_w_half,
                                     cv2.BORDER_CONSTANT, value=self.padding)
        img_height, img_width = img.shape[0:2]
        if self.crop_type == 'rand':
            h_off = random.randint(0, img_height - self.crop_H)
            w_off = random.randint(0, img_width - self.crop_W)
        else:
            h_off = (img_height - self.crop_H) / 2
            w_off = (img_width - self.crop_W) / 2
        img = img[h_off:h_off + self.crop_H, w_off:w_off + self.crop_W]
        return img


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given ndarray image (H*W*C or H*W) and label with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        return img


class RandomVerticalFlip(object):
    """
    Randomly vertically flips the given ndarray image (H*W*C or H*W) and label with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
        return img


class RandScale(object):
    """
    Randomly resize image & seglabel with scale factor in [scale_min, scale_max]
    """

    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("imgseg_transforms.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("imgseg_transforms.RandScale() aspect_ratio param error.\n"))

    def __call__(self, img):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        img = cv2.resize(img, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        return img


class ColorJitter(object):
    """
    do ColorJitter for image & seglabel
    factor should be a number of list of three number, all numbers should be in (0,1)
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        tool = ColorJitterImg(self.factor)
        img = tool(img)
        return img
