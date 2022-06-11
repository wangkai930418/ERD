#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0  python test_FT_cifar.py \
                                --gpu 0 \
                                --dataset CIFAR100 \
                                --way 5 \
                                --shot 1 \
                                --lr 0.001 \
                                --step_size 50 \
                                --batch_size 50 \
                                --TASK_NUM 16 \
                                --BASE_CLS_NUM 5 \
                                --gamma 0.8 \
                                --model convnet \
                                --dim 1600 \
                                --method ft \
                                --max_epoch 200

CUDA_VISIBLE_DEVICES=0  python test_ERD_cifar.py \
                                --gpu 0 \
                                --dataset CIFAR100 \
                                --way 5 \
                                --shot 1 \
                                --lr 0.001 \
                                --step_size 50 \
                                --batch_size 50 \
                                --TASK_NUM 16 \
                                --BASE_CLS_NUM 5 \
                                --gamma 0.8 \
                                --model convnet \
                                --dim 1600 \
                                --method ft \
                                --max_epoch 200