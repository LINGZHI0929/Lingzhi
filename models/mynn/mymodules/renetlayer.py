import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['ReNetLayer']


class ReNetLayer(nn.Module):
    def __init__(self, inchannel, hiddenchannel, direction):
        super(ReNetLayer, self).__init__()
        self.inchannel = inchannel
        # real output channel will be self.hiddenchannel * 2, concat result of 2 directions
        self.hiddenchannel = hiddenchannel
        self.lstm1 = nn.LSTM(self.inchannel, self.hiddenchannel)
        self.lstm2 = nn.LSTM(self.inchannel, self.hiddenchannel)
        if direction == "width":
            self.direction = 0
        elif direction == "height":
            self.direction = 1
        else:
            raise ValueError("direction for ReNetLayer should be \"width\" | \"height\"\n")

    def forward(self, inputs):
        if inputs.dim() != 4 and inputs.dim != 3:
            raise RuntimeError("Input of ReNetLayer must be NxCxHxW or CxHxW\n")
        if inputs.size(-3) != self.inchannel:
            raise RuntimeError("Input channel of ReNetLayer error: {} vs {}\n".format(inputs.size(-3), self.inchannel))
        W = inputs.size(-1)
        H = inputs.size(-2)
        C = inputs.size(-3)
        only3dim = inputs.dim == 3

        # along width
        # input: NxCxHxW --> Wx(NxH)xC | CxHxW --> WxHxC
        # h_0  : 1 * NxH * hiddenchannel | 1 * H * hiddenchannel
        # c_0  : 1 * NxH * hiddenchannel | 1 * H * hiddenchannel
        if self.direction == 0:
            if only3dim:
                inputs = inputs.permute(2, 1, 0)
            else:
                inputs = inputs.permute(3, 0, 2, 1).contiguous().view(W, -1, C)
        # along height
        # input: NxCxHxW --> Hx(NxW)xC | CxHxW --> HxWxC
        # h_0  : 1 * NxW * hiddenchannel | 1 * W * hiddenchannel
        # c_0  : 1 * NxW * hiddenchannel | 1 * W * hiddenchannel
        else:
            if only3dim:
                inputs = inputs.permute(1, 2, 0)
            else:
                inputs = inputs.permute(2, 0, 3, 1).contiguous().view(H, -1, C)

        inv_idx = Variable(torch.arange(inputs.size(0) - 1, -1, -1).type_as(inputs.data).long())
        inputs_inv = inputs.index_select(0, inv_idx)

        init_hidden = (Variable(torch.zeros(1, inputs.size(1), self.hiddenchannel).type_as(inputs.data)),
                       Variable(torch.zeros(1, inputs.size(1), self.hiddenchannel).type_as(inputs.data)))

        outputs, _ = self.lstm1(inputs, init_hidden)
        outputs_inv, _ = self.lstm2(inputs_inv, init_hidden)
        outputs_inv = outputs_inv.index_select(0, inv_idx)

        if self.direction == 0:
            if only3dim == 3:
                outputs = outputs.permute(2, 1, 0)
                outputs_inv = outputs_inv.permute(2, 1, 0)
            else:
                outputs = outputs.contiguous().view(W, -1, H, self.hiddenchannel).permute(1, 3, 2, 0)
                outputs_inv = outputs_inv.contiguous().view(W, -1, H, self.hiddenchannel).permute(1, 3, 2, 0)
        else:
            if only3dim == 3:
                outputs = outputs.permute(2, 0, 1)
                outputs_inv = outputs_inv.permute(2, 0, 1)
            else:
                outputs = outputs.contiguous().view(H, -1, W, self.hiddenchannel).permute(1, 3, 0, 2)
                outputs_inv = outputs_inv.contiguous().view(H, -1, W, self.hiddenchannel).permute(1, 3, 0, 2)

        return torch.cat((outputs, outputs_inv), 1)
