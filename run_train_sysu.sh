#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/SYSU-MM01}
GPU=${CUDA_VISIBLE_DEVICES:-0,1}

if [[ "$DATA_DIR" != /* ]]; then
  printf '\033[31mWARNING: 您使用的是相对路径 --data-dir "%s"；请从仓库根目录运行，或确认数据已放在该相对路径下。\033[0m\n' "$DATA_DIR"
fi

CUDA_VISIBLE_DEVICES=$GPU \
python train_sysu.py -mb CMhybrid --epochs 50 -b 128 -a agw -d sysu_all \
--iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 \
--data-dir "$DATA_DIR"

