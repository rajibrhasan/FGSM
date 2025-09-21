DATASET="mnist"
MODEL="resnet18"
CHECKPOINT_DIR="checkpoints/${DATASET}/${MODEL}"

mkdir -p $CHECKPOINT_DIR

python train.py \
    --dir data \
    --dataset $DATASET \
    --model $MODEL \
    --checkpoint_dir $CHECKPOINT_DIR \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --num_classes 10