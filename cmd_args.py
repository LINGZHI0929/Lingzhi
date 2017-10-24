import argparse

parser = argparse.ArgumentParser(description='PyTorch Image Segmentation Training')

# DNN structure
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet50_dilated_forseg_x8',
                    help='backbone model architecture (default: resnet50_dilated_forseg_x8)')
parser.add_argument('--net',
                    metavar='NET',
                    default='Seg',
                    choices=['Raw', 'Seg', 'ReSeg', 'ReSegRes'],
                    help='model type (default: Seg)')
parser.add_argument('--classes',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of segmentation classes (default: 2)')
parser.add_argument('--fchannel',
                    default=0,
                    type=int,
                    metavar='N',
                    help='number of feature channel (default: 0)')
parser.add_argument('--zoom_factor',
                    default=8,
                    type=int,
                    metavar='N',
                    help='zoom factor (default: 8)')
parser.add_argument('--dropout',
                    default=0.5,
                    type=float,
                    metavar='Dropout',
                    help='dropout ratio of feature')
parser.add_argument('--hiddenchannel',
                    default=128,
                    type=int,
                    metavar='N',
                    help='number of hidden channel for LSTM in ReNetLayer (default: 128)')
parser.add_argument('--renet_num',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of ReNetLayer for ReSeg (default: 1)')

# dataset
parser.add_argument('--dataroot',
                    default='',
                    type=str,
                    metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--trainlist',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to img&seg TRAIN pair list file')
parser.add_argument('--vallist',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to img&seg VALIDATION pair list file')
parser.add_argument('--testlist',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to TEST image list file (no label)')
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 8)')

# step learning rate
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--step_epoch',
                    default=10,
                    type=int,
                    metavar='N',
                    help='step learning rate strategy')

# other args
parser.add_argument('--print_freq',
                    default=5,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--saveprefix',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='prefix of saved checkpoint (default: none)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

# for output test result
parser.add_argument('--savedir',
                    default='',
                    type=str,
                    metavar='DIR',
                    help='path to save root')
parser.add_argument('--savedir_prob',
                    default='',
                    type=str,
                    metavar='DIR',
                    help='path to save root')
