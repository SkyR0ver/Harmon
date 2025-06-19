#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH

GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=2,3 \
    torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ../scripts/train.py \
    ../configs/examples/qwen2_5_0_5b_kl16_mar_b_train.py \
    --launcher pytorch \
    --deepspeed deepspeed_zero2
