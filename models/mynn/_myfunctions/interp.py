from numbers import Integral

from torch.autograd import Function
from torch._thnn import type2backend

__all__ = ['InterpBilinear2d']


class _InterpBase(Function):

    def __init__(self, size=None, zoom_factor=None, shrink_factor=None):
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
        self.input_size = None
        self.output_size = None


class InterpBilinear2d(_InterpBase):
    def forward(self, input):
        assert input.dim() == 4
        self.input_size = input.size()

        if self.size:
            self.output_size = self.size
        elif self.zoom_factor:
            self.output_size = (
                (input.size(2) - 1) * self.zoom_factor + 1,
                (input.size(3) - 1) * self.zoom_factor + 1,
            )
        elif self.shrink_factor:
            self.output_size = (
                (input.size(2) - 1) / self.shrink_factor + 1,
                (input.size(3) - 1) / self.shrink_factor + 1,
            )
        else:
            raise ValueError("Interp need size, zoom_factor or shrink_factor.\n")

        output = input.new()
        backend = type2backend[type(input)]
        backend.SpatialUpSamplingBilinear_updateOutput(
            backend.library_state,
            input,
            output,
            self.output_size[0],
            self.output_size[1],
        )
        return output

    def backward(self, grad_output):
        assert grad_output.dim() == 4

        grad_output = grad_output.contiguous()
        grad_input = grad_output.new()
        backend = type2backend[type(grad_output)]
        backend.SpatialUpSamplingBilinear_updateGradInput(
            backend.library_state,
            grad_output,
            grad_input,
            self.input_size[0],
            self.input_size[1],
            self.input_size[2],
            self.input_size[3],
            self.output_size[0],
            self.output_size[1],
        )
        return grad_input
