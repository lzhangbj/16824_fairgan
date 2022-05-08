#!/usr/bin/env bash



cp detect_bboxes.py /home/linz/datasets/FairFace/

ln -s /home/linz/datasets/FairFace /home/linz/code/imbalanced-face/datasets/

cd /home/linz/code/imbalanced-face/datasets/FairFace

python detect_bboxes.py