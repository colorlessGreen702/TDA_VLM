#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --datasets caltech101/dtd/oxford_pets/ucf101/A/V\
                                                --backbone ViT-B/16