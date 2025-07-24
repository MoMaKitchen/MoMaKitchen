source /cpfs01/shared/optimal/zhangpingrui/.bashrc
conda activate python38

SAVE_DIR="runs"
TRAINING_NAME="train"

mkdir -p "$SAVE_DIR/$TRAINING_NAME"

export CUDA_VISIBLE_DEVICES=1

python -u train.py \
save_dir=$SAVE_DIR \
name=$TRAINING_NAME 2>&1 \