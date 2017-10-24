#!/bin/sh

python main.py \
  --dataroot=exp_example \
  --trainlist=exp_example/list.txt \
  --vallist=exp_example/list.txt \
  --arch=resnet50_dilated_forseg_x8 \
  --net=Seg \
  --classes=5 \
  --fchannel=128 \
  --zoom_factor=8 \
  --workers=1 \
  --batch_size=1 \
  --lr=0.01 \
  --epochs=40 \
  --step_epoch=10 \
  --print_freq=1 \
  --evaluate \
  --saveprefix=exp_example/try