#!/usr/bin/env bash



cp detect_bboxes.py /home/linz/datasets/FakeMake/

ln -s /home/linz/datasets/FakeMake /home/linz/code/imbalanced-face/datasets/

cd /home/linz/code/imbalanced-face/datasets/FakeMake

python detect_bboxes.py