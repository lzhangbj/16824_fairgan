#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
MASTER_ADDR="128.2.205.94"
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NCCL_DEBUG=WARN OMP_NUM_THREADS=8 NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=mlx5_2 NCCL_SOCKET_IFNAME=eth0 \
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    --nnodes=1 --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=${PORT} \
    tools/test.py $CONFIG --launcher pytorch ${@:3}