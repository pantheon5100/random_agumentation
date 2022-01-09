python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 100 \
    --lr-schedule multistep \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 4 \
    --attack-iters 7 \
    --delta-init random \
    --out-dir PGD_baseline \
    --seed 0 \
    --image_normalize

