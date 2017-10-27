import os
import time
import logging
import collections

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import getmodel
from models.mynn import CrossEntropy2d
import datasets
from datasets import imgseg_transforms
from util import ConfusionMeter
from cmd_args import parser

# logger
logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)


def main():
    global args
    args = parser.parse_args()
    ignore_label = 255

    # create model
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("   ARCH: {}".format(args.arch))
    logger.info("    NET: {}".format(args.net))
    logger.info("Classes: {}".format(args.classes))
    model = getmodel.GetModel(args)
    print(model)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.enabled = True
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = CrossEntropy2d(size_average=True, ignore_label=ignore_label).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    mean = [103.939, 116.779, 123.68]
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageSeg(args.dataroot, args.trainlist, imgseg_transforms.Compose([
            imgseg_transforms.RandScale([0.95, 1.05], aspect_ratio=[0.95, 1.05]),
            imgseg_transforms.ColorJitter(0.3),
            imgseg_transforms.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=ignore_label),
            imgseg_transforms.ToTensor(),
            imgseg_transforms.Normalize(mean=mean)
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = None
    if args.evaluate:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageSeg(args.dataroot, args.vallist, imgseg_transforms.Compose([
                imgseg_transforms.Resize([713, 713]),
                imgseg_transforms.ToTensor(),
                imgseg_transforms.Normalize(mean=mean)
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # save checkpoint
        filename = args.saveprefix + '_epoch_' + str(epoch) + '.checkpoint.pth.tar'
        logger.info('Saving checkpoint to: ' + filename)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)
        # evaluate on validation set
        if args.evaluate:
            validate(val_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # delete loss & output for saving GPU memory (scoping rule of python)
        del output, loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'DataTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f}'.format(epoch, i + 1, len(train_loader),
                                                     batch_time=batch_time,
                                                     data_time=data_time,
                                                     loss=losses))


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    forward_time = AverageMeter()
    losses = AverageMeter()
    confusion_mat_all = ConfusionMeter(args.classes)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model(input_var)
        loss = criterion(output, target_var)

        # forward time
        forward_time.update(time.time() - end)

        # measure accuracy and record loss
        confusion_mat_all.add(output.data, target)
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'ForwardTime {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                        'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f}'.format(i + 1, len(val_loader),
                                                     forward_time=forward_time,
                                                     batch_time=batch_time,
                                                     loss=losses))

    result, classresult = accuarcy(confusion_mat_all.value(), classlist=[0, 1, 2, 3, 4])
    logger.info('Evaluation Result:\t'
                'Precision  {0:.4f}\t'
                'meanRecall {1:.4f}\t'
                'meanIOU {2:.4f}'.format(result[0], result[1], result[2]))
    for i in range(len(classresult)):
        logger.info('Class {0}  Result:\t'
                    'Precision  {1:.4f}\t'
                    'Recall     {2:.4f}\t'
                    'IOU        {3:.4f}'.format(i, classresult[i][0], classresult[i][1], classresult[i][2]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuarcy(conf_mat, classlist=None):
    """Computes the Segmentation Accuracy"""
    assert conf_mat.dim() == 2
    assert conf_mat.size(0) == conf_mat.size(1) == args.classes
    acc = sum(conf_mat.diag()) / sum(sum(conf_mat))
    recall = conf_mat.diag().view(-1) / conf_mat.sum(1).view(-1)
    precision = conf_mat.diag().view(-1) / conf_mat.sum(0).view(-1)
    iou = conf_mat.diag().view(-1) / (conf_mat.sum(1).view(-1) + conf_mat.sum(0).view(-1) - conf_mat.diag().view(-1))
    totalresult = (acc, recall.mean(), iou.mean())
    classresult = list()
    if classlist is not None and isinstance(classlist, collections.Iterable):
        for i in classlist:
            assert (isinstance(i, int) and 0 <= i < args.classes)
            classresult.append((precision[i], recall[i], iou[i]))
    return totalresult, classresult


if __name__ == '__main__':
    main()
