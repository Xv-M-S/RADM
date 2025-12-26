#!/bin/bash

# RADM+RL(DDPO) 训练脚本
# 在TASI-SERM分支基础上集成DDPO优化

echo "Starting RADM+DDPO training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# DDPO训练配置
CONFIG_FILE="configs/radm.yaml"
NUM_GPUS=1
OUTPUT_DIR="./output_ddpo"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练命令
python train_net.py \
    --num-gpus $NUM_GPUS \
    --config-file $CONFIG_FILE \
    --resume \
    2>&1 | tee $OUTPUT_DIR/train_log.txt

echo "Training completed. Logs saved to $OUTPUT_DIR/train_log.txt"

