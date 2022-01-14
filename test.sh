#!/bin/bash

python train.py --dataset ours --max_epoch 2 --batch_size 2 --N 5 --K 3 --Q 3 --metric micro_f1 --lr 2e-5  >>log/2_2_533.log 2>&1

python train.py --dataset ours --max_epoch 1 --batch_size 2 --N 5 --K 1 --Q 1 --metric micro_f1 --lr 2e-5  >>log/1_2_511.log 2>&1
python train.py --dataset ours --max_epoch 2 --batch_size 2 --N 5 --K 1 --Q 1 --metric micro_f1 --lr 2e-5  >>log/2_2_511.log 2>&1

python train.py --dataset ours --max_epoch 1 --batch_size 2 --N 3 --K 3 --Q 3 --metric micro_f1 --lr 2e-5  >>log/1_2_333.log 2>&1
python train.py --dataset ours --max_epoch 2 --batch_size 2 --N 3 --K 3 --Q 3 --metric micro_f1 --lr 2e-5  >>log/2_2_333.log 2>&1

python train.py --dataset ours --max_epoch 1 --batch_size 2 --N 3 --K 1 --Q 1 --metric micro_f1 --lr 2e-5  >>log/1_2_311.log 2>&1
python train.py --dataset ours --max_epoch 2 --batch_size 2 --N 3 --K 1 --Q 1 --metric micro_f1 --lr 2e-5  >>log/2_2_311.log 2>&1




