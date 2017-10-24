#!/bin/sh

python main_cs.py \
  --dataroot=/home/zhangyi/Cityscapes \
  --trainlist=/home/zhangyi/Cityscapes/filelist/fine_train.txt \
  --arch=resnet50_dilated_forseg_x8 \
  --net=Seg \
  --classes=19 \
  --fchannel=128 \
  --zoom_factor=8 \
  --workers=4 \
  --batch_size=4 \
  --lr=0.01 \
  --epochs=40 \
  --step_epoch=10 \
  --print_freq=10 \
  --saveprefix=cs_exp/res50/model
