import numpy as np
import numbers
import collections
import random


def grayscale(pic):
    assert (len(pic.shape) == 3) and (pic.shape[2] == 3), "input img for grayscale() should be H*W*3 ndarray"
    grayimg = 0.299 * pic[:, :, 2] + 0.587 * pic[:, :, 1] + 0.114 * pic[:, :, 0]
    grayimg = np.repeat(grayimg[:, :, np.newaxis], 3, axis=2)
    return grayimg


def saturation(pic, factor):
    grayimg = grayscale(pic)
    img = pic * factor + grayimg * (1.0 - factor)
    img = img.astype(pic.dtype)
    return img


def brightness(pic, factor):
    img = pic * factor
    return img


def contrast(pic, factor):
    grayimg = grayscale(pic)
    ave = grayimg[:, :, 0].mean()
    ave_img = np.ndarray(shape=pic.shape, dtype=float)
    ave_img.fill(ave)
    img = pic * factor + ave_img * (1 - factor)
    return img


class ColorJitterImg(object):
    """
    do ColorJitter for BGR ndarray image
    factor should be a number of list of three number, all numbers should be in (0,1)
    """

    def __init__(self, factor):
        if isinstance(factor, numbers.Number) and 0 < factor < 1:
            self.saturation_factor = factor
            self.brightness_factor = factor
            self.contrast_factor = factor
        elif isinstance(factor, collections.Iterable) and len(factor) == 3 \
                and isinstance(factor[0], numbers.Number) and 0 < factor[0] < 1 \
                and isinstance(factor[1], numbers.Number) and 0 < factor[1] < 1 \
                and isinstance(factor[2], numbers.Number) and 0 < factor[2] < 1:
            self.saturation_factor = factor[0]
            self.brightness_factor = factor[1]
            self.contrast_factor = factor[2]
        else:
            raise (RuntimeError("ColorJitter factor error.\n"))

    def __call__(self, img):
        ori_type = img.dtype
        img.astype('float32')
        this_saturation_factor = 1.0 + self.saturation_factor * random.uniform(-1.0, 1.0)
        this_brightness_factor = 1.0 + self.brightness_factor * random.uniform(-1.0, 1.0)
        this_contrast_factor = 1.0 + self.contrast_factor * random.uniform(-1.0, 1.0)
        funclist = [(saturation, this_saturation_factor),
                    (brightness, this_brightness_factor),
                    (contrast, this_contrast_factor)]
        random.shuffle(funclist)
        for func in funclist:
            img = (func[0])(img, func[1])
        if ori_type == np.uint8:
            img = np.clip(img, 0, 255)
            img.astype('uint8')
        elif ori_type == np.uint16:
            img = np.clip(img, 0, 65535)
            img.astype('uint16')
        return img
