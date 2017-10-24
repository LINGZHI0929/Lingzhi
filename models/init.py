import torch.nn as nn
import torch.nn.init as initer


def InitModel(model, conv='kaiming', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conv == 'kaiming':
                initer.kaiming_normal(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            initer.constant(m.weight, 1)
            initer.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant(param, 0)
