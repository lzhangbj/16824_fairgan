#!/usr/bin/env bash

work_dir=$1

python tools/produce_importance.py \
    configs/recognition/classification/fairface/hyperstyle_4_races_025/race_cls_fairface_4race025-7000-white-5380-540-balanced_resnet34_adam.py \
    --work-dir=${work_dir} \
    --test_data_root=./datasets/FairFace/025_images \
    --load_from=${work_dir}/best_model.pth \
    --save_name=embeddings_importance_v2 \
    --method=v1