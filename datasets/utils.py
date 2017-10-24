import os
import os.path
import hashlib
import errno
import torch
import math


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W),
    or a list of images all of the same size,
    makes a grid of images of size (B / nrow, nrow).

    normalize=True will shift the image to the range (0, 1),
    by subtracting the minimum and dividing by the maximum pixel value.

    if range=(min, max) where min and max are numbers, then these numbers are used to
    normalize the image.

    scale_each=True will scale each image in the batch of images separately rather than
    computing the (min, max) over all images.

    pad_value=<float> sets the value for the padded pixels.

    [Example usage is given in this notebook](https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91)
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensorlist = tensor
        numImages = len(tensorlist)
        size = torch.Size(torch.Size([numImages]) + tensorlist[0].size())
        tensor = tensorlist[0].new(size)
        for i in range(numImages):
            tensor[i].copy_(tensorlist[i])

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min_v, max_v):
            img.clamp_(min=min_v, max=max_v)
            img.add_(-min_v).div_(max_v - min_v)

        def norm_range(ts, v_range):
            if v_range is not None:
                norm_ip(ts, v_range[0], v_range[1])
            else:
                norm_ip(ts, ts.min(), ts.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + 1 + padding // 2, height - padding)\
                .narrow(2, x * width + 1 + padding // 2, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, value_range=None, scale_each=False, pad_value=0):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images by calling `make_grid`.
    All options after `filename` are passed through to `make_grid`. Refer to it's documentation for
    more details
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, value_range=value_range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
