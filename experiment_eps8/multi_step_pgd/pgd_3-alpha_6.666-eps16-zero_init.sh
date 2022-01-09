python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 6.6666 \
    --attack-iters 3 \
    --delta-init zero \
    --out-dir PGD_baseline \
    --seed 1 \
    --image_normalize

