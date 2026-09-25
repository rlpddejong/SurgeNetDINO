#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of Nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=24:00:00                         # Specify the maximum time the job can run

# Load Conda environment
source "<path/to/miniconda3>/etc/profile.d/conda.sh"
conda activate "<your_conda_env>"

# Specify global variables
OUTPUT_FOLDER="<path/to/output_folder>"
DATA_PATH="<path/to/data>"  # folder with seg_videos / seg_video_masks / MTL_split
NUM_DEVICES=1

# Weights & Biases configuration
export WANDB_API_KEY="<your_wandb_api_key>"
export WANDB_DIR=$OUTPUT_FOLDER/wandb
export WANDB_CONFIG_DIR=$OUTPUT_FOLDER/wandb
export WANDB_CACHE_DIR=$OUTPUT_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

# Model = config name in configs/, e.g. sbatch train_eomt.sh eomt_small_dinov3_336_SurgeNetXL
# eomt_{small,base}_dinov1 | eomt_{small,base,large}_dinov2 | eomt_{small,base,large}_dinov3 (+ _336_SurgeNetXL)
EXPERIMENT_NAME=${1:-eomt_large_dinov2_336_SurgeNetXL}
python3 main.py fit \
  -c configs/$EXPERIMENT_NAME.yaml \
  --data.path $DATA_PATH \
  --trainer.devices $NUM_DEVICES \
  --trainer.logger.project $EXPERIMENT_NAME \
  --trainer.default_root_dir $DATA_PATH/results/$EXPERIMENT_NAME \
  --trainer.logger.save_dir $DATA_PATH/results/$EXPERIMENT_NAME
