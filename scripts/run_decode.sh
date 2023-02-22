#!/usr/bin/env bash

# Resolve CPU and GPU
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
TOTAL_GPUS=`nvidia-smi -L | wc -l`
OMP_NUM_THREADS=`expr $TOTAL_CORES / $TOTAL_GPUS`

# If you want to specify gpus, uncomment and edit below three lines.
#TOTAL_GPUS=3
#CUDA_VISIBLE_DEVICES=0,1,3 \
#OMP_NUM_THREADS=$OMP_NUM_THREADS \
python -u run_decode.py --nproc_per_node=${TOTAL_GPUS} \
--model_dir diffusion_models/{name-of-model-folder} \
--sample_seed 123
