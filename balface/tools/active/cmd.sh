bash tools/active/active_run.sh \
  4race_imb_14k_aug_40k_sample_1000 \
  5 1000


CUDA_VISIBLE_DEVICES=2 python generate.py --network=./pretrained_models/stylegan2_fairface025_full_eric.pkl \
    --rand_start=20000 \
    --num-images=10000 \
    --outdir=stylegan2_generated_10k

python tools/compute_importance.py \
  --ckpt=./work_dirs/eval_model.pth  \
  --image_dir=./datasets/FairFace/025_images/old/4race_imb_14k_aug_40k_rand-sample_1000_stage5 \
  --save-name=./embeddings_scores/fairface_race_imb_14k_aug_40k_rand-sample_1000_stage5

cd datasets/UTKFace
rm -rf labels
rclone copy -P gdrive:datasets/UTKFace/labels.zip ./
unzip -q labels.zip
rm labels.zip


rclone copy -P ./datasets/FairFace/labels/mix-aug/race_cutmix_intra_imb-14k_x1.txt gdrive:datasets/fairface/setup/labels/mix-aug/
cd ./datasets/FairFace/025_images/
zip -q -r race_cutmix_intra_imb-14k_x1.zip race_cutmix_intra_imb-14k_x1
rclone copy -P race_cutmix_intra_imb-14k_x1.zip gdrive:datasets/fairface/mix-data/


rclone copy -P gdrive:datasets/fairface/setup/labels/mix-aug/race_cutmix_intra_imb-14k_x1.txt ./datasets/FairFace/labels/mix-aug/
rclone copy -P gdrive:datasets/fairface/mix-data/race_cutmix_intra_imb-14k_x1.zip ./datasets/FairFace/025_images/
cd ./datasets/FairFace/025_images/
unzip -q race_cutmix_intra_imb-14k_x1.zip
wait
rm race_cutmix_intra_imb-14k_x1.zip
cd -
