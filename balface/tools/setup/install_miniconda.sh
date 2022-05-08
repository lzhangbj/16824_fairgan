#!/usr/bin/env bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
wait
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh

export PATH=~/miniconda/bin:$PATH