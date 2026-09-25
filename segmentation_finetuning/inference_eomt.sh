#!/bin/bash

# Load Conda environment
source "<path/to/miniconda3>/etc/profile.d/conda.sh"
conda activate "<your_conda_env>"

# Specify global variables
DATA_PATH="<path/to/data>"  # folder with seg_videos / seg_video_masks / MTL_split

# Model = config name in configs/ (same as for train_eomt.sh)
EXPERIMENT_NAME=${1:-eomt_large_dinov2_336_SurgeNetXL}
RESULTS_DIR=$DATA_PATH/results/$EXPERIMENT_NAME

# Optional positional arguments:
#   $1  model / config name    (default: eomt_large_dinov2_336_SurgeNetXL)
#   $2  checkpoint path        (default: latest best-*.ckpt found under RESULTS_DIR)
#   $3  number of last frames  (default: data.init_args.num_predict_frames in config)
#   $4  number of video clips  (default: data.init_args.num_video_clips in config)
#   $5  metrics_only           (any non-empty value => only (re)compute metrics
#                               from already-saved frames, no inference)
CKPT_PATH=${2:-""}
NUM_PREDICT_FRAMES=${3:-""}
NUM_VIDEO_CLIPS=${4:-""}
METRICS_ONLY=${5:-""}

python3 inference_eomt.py \
  --config configs/$EXPERIMENT_NAME.yaml \
  --data_path "$DATA_PATH" \
  --save_dir "$RESULTS_DIR" \
  ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
  ${NUM_PREDICT_FRAMES:+--num_predict_frames "$NUM_PREDICT_FRAMES"} \
  ${NUM_VIDEO_CLIPS:+--num_video_clips "$NUM_VIDEO_CLIPS"} \
  ${METRICS_ONLY:+--metrics_only}
