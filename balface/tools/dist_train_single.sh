#!/usr/bin/env bash

# bash tools/dist_train_single.sh {config_file} {num_gpus}

CONFIG=$1
GPUS=$2
MASTER_ADDR="10.22.152.77"
PORT=${PORT:-29500}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    --nnodes=1 --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=${PORT} \
    tools/train.py $CONFIG --launcher pytorch ${@:3}
