DATASET="mnist"
MODEL="resnet18"
CHECKPOINT_DIR="checkpoints/${DATASET}/${MODEL}"
LOGS_DIR="logs"

mkdir -p $LOGS_DIR
python fgsm_attack.py \
    --dir data \
    --dataset $DATASET \
    --model $MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --batch_size 64 \
    --num_classes 10 \
    --log_dir $LOGS_DIR