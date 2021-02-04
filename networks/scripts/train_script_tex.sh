#!/usr/bin/env bash
set -ex
GPU_ID=0
NAME='pamir_texture'
USE_ADAPTIVE_GEO_LOSS='False'
USE_GT_SMPL_VOLUME='True'
USE_MULTISTAGE_LOSS='True'
PRETRAINED_GCMR_CHECKPOINT='./results/gcmr_pretrained'
PRETRAINED_PAMIR_NET_CHECKPOINT='./results/pamir_geometry/checkpoints/latest.pt'
LR=2e-4
BATCH_SIZE=3
LOG_DIR='./results'
DATASET_DIR='../dataset_example'
VIEW_NUM_PER_ITEM=360
POINT_NUM=5000
NUM_EPOCHS=2
SUMMARY_STEPS=20
CHECKPOINTS_STEPS=20000
TEST_STEPS=5000
NUM_WORKERS=12

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_train_tex.py \
--name ${NAME} \
--log_dir ${LOG_DIR} \
--pretrained_gcmr_checkpoint ${PRETRAINED_GCMR_CHECKPOINT} \
--pretrained_pamir_net_checkpoint ${PRETRAINED_PAMIR_NET_CHECKPOINT} \
--dataset_dir ${DATASET_DIR} \
--view_num_per_item ${VIEW_NUM_PER_ITEM} \
--num_epochs ${NUM_EPOCHS} \
--use_adaptive_geo_loss ${USE_ADAPTIVE_GEO_LOSS} \
--use_gt_smpl_volume ${USE_GT_SMPL_VOLUME} \
--use_multistage_loss ${USE_MULTISTAGE_LOSS} \
--lr ${LR} \
--point_num ${POINT_NUM} \
--batch_size ${BATCH_SIZE} \
--summary_steps ${SUMMARY_STEPS} \
--checkpoint_steps ${CHECKPOINTS_STEPS} \
--test_steps ${TEST_STEPS} \
--num_workers ${NUM_WORKERS} \
--shuffle_train --debug
