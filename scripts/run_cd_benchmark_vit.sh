#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets caltech101/dtd/oxford_pets/ucf101/A/R\
                                                --backbone ViT-B/16