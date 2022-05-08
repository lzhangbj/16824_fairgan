#!/usr/bin/env bash

ORIG_POOL_ROOT=$1
NUM_STAGE=$2
SAMPLE_NUM=$3
CKPT_FILE='./work_dirs/race_cls_fairface_4race025-14000-white-10760-1080-balanced_resnet34_adam/epoch_50.pth'

#POOL_ROOT=$ORIG_POOL_ROOT

for STAGE in $(seq 1 $NUM_STAGE)
do
  echo "##################### stage {$STAGE} ####################"
  POOL_ROOT=${ORIG_POOL_ROOT}_stage${STAGE}
#
#  if [ $STAGE -gt 1 ]
#  then
  # produce importance file and folder
  python tools/active/compute_importance_v2.py \
    --image_dir=./datasets/FairFace/025_images/${ORIG_POOL_ROOT} \
    --save_dir=./datasets/FairFace/025_images/${POOL_ROOT} \
    --ckpt=${CKPT_FILE}

  # select and copy image from src folder into importance folder
  # produce label file for it
  python tools/active/sample_images.py \
    --src-root=./datasets/FairFace/025_images/${ORIG_POOL_ROOT} \
    --dest-root=./datasets/FairFace/025_images/${POOL_ROOT} \
    --dest-label-file=./datasets/FairFace/labels/cond-aug/${POOL_ROOT}.txt \
    --importance-file=./datasets/FairFace/025_images/${POOL_ROOT}/scores.npy \
    --sample-method=V1 \
    --sample-num=${SAMPLE_NUM}

  bash tools/dist_train.sh configs/active/${ORIG_POOL_ROOT}/${POOL_ROOT}.py 8
#  fi

  CKPT_FILE=./work_dirs/${POOL_ROOT}/best_model.pth

done