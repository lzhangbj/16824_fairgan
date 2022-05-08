#!/usr/bin/env bash

cd ../balface
#export https_proxy="http://bj-rd-proxy.byted.org:3128"
conda create -n balface python==3.8
conda activate balface
conda install -y pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
#sudo apt-get install ffmpeg libsm6 libxext6 -y