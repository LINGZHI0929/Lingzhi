from numbers import Integral

import torch.nn as nn
from .. import myfunctional as myF


class _InterpBase(nn.Module):
    def __init__(self, size=None, zoom_factor=None, shrink_factor=None):
        # type: (object, object, object) -> object
        super(_InterpBase, self).__init__()
        self.size = None
        self.zoom_factor = None
        self.shrink_factor = None
        if size is not None:
            if isinstance(size, Integral) and size > 0:
                self.size = (size, size)
            elif isinstance(size, tuple) and len(size) == 2 \
                    and isinstance(size[0], Integral) and size[0] > 0 \
                    and isinstance(size[1], Integral) and size[1] > 0:
                self.size = size  # (height, width)
            else:
                raise ValueError("size for Interp error: {}\n".format(size))
        elif zoom_factor is not None:
            if isinstance(zoom_factor, Integral) and zoom_factor > 1:
                self.zoom_factor = zoom_factor
            else:
                raise ValueError("zoom_factor for Interp error: {}\n".format(zoom_factor))
        elif shrink_factor is not None:
            if isinstance(shrink_factor, Integral) and shrink_factor > 1:
                self.shrink_factor = shrink_factor
            else:
                raise ValueError("shrink_factor for Interp error: {}".format(shrink_factor))
        else:
            raise ValueError("Interp need size, zoom_factor or shrink_factor.\n")

    def __repr__(self):
        if self.size:
            info = 'size=' + str(self.size)
        elif self.zoom_factor:
            info = 'zoom_factor=' + str(self.zoom_factor)
        elif self.shrink_factor:
            info = 'shrink_factor=' + str(self.shrink_factor)
        else:
            info = 'no size, zoom_factor or shrink_factor'
        return self.__class__.__name__ + '(' + info + ')'

    def forward(self, *input):
        raise RuntimeError("Abstract method of base class should be overrided\n")


class InterpBilinear2d(_InterpBase):
    def forward(self, input):
        return myF.interp_bilinear(input, self.size, self.zoom_factor, self.shrink_factor)
