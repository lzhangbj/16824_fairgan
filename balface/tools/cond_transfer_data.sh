#!/usr/bin/env bash

name=$1

rclone copy -P gdrive:datasets/fairface/setup/labels/cond-aug/$name.txt ./datasets/FairFace/labels/cond-aug/
cd datasets/FairFace/025_images
hdfs dfs -get /home/byte_arnold_hl_vc/zhanglin.linz/dataset/$name.zip ./
unzip -q $name.zip
rm *.zip -rf
cd ../../..