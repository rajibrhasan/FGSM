DATASET="cifar10"
MODEL="vit_tiny_patch16_224"
CHECKPOINT_DIR="checkpoints/${DATASET}/${MODEL}"
LOGS_DIR="logs"

mkdir -p $LOGS_DIR

python fgsm_attack.py \
    --dir data \
    --dataset $DATASET \
    --model $MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --batch_size 128 \
    --num_classes 10