import torch

__all__ = ['ConfusionMeter']


class ConfusionMeter(object):
    """
    The `confusionmeter.ConfusionMeter` constructs a confusion matrix for a multi-class
    classification problems.

    At initialization time, the `k` parameter that indicates the number of classes in the
    classification problem under consideration must be specified.

    The `add(output, target)` method takes as input an NxK (or NxKxHxW) tensor `output`
    that contains the output scores obtained from the model for N examples and K classes
    (H, W is height and width), and a corresponding N-tensor (or NxHxW-tensor) `target`
    that provides the targets for the N examples (H, W is height and width). The targets
    are assumed to be integer values between 0 and K-1.

    The `value(normalized = False)` method returns the confusion matrix in a KxK tensor.
    In the confusion matrix, rows correspond to ground-truth targets and columns correspond
    to predicted targets. Parameter `normalized` (default = `false`) may be specified that
    determines whether or not the confusion matrix is normalized or not when calling value()
    """

    def __init__(self, k):
        super(ConfusionMeter, self).__init__()
        self.classes = k
        self.conf = torch.zeros(k, k)

    def reset(self):
        self.conf.fill_(0)

    def add(self, output, target):

        assert (output.dim() == 4 or output.dim() == 2)
        assert target.dim() == output.dim() - 1
        assert output.size(0) == target.size(0)
        assert output.size(1) == self.classes
        if output.dim() == 4:
            assert output.size(2) == target.size(1)
            assert output.size(3) == target.size(2)
        _, pr = output.max(1)
        pr = torch.squeeze(pr, 1)
        gt = target.int()
        pr = pr.int()

        for gt_i in range(self.classes):
            for pr_i in range(self.classes):
                num = (gt == gt_i) * (pr == pr_i)
                self.conf[gt_i][pr_i] += num.sum()

    def value(self, normalized=False):
        if normalized:
            return self.conf / self.conf.sum(1).clamp_(min=0.001).repeat(1, self.classes)
        else:
            return self.conf
