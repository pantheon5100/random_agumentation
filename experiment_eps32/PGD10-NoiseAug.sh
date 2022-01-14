python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 32 \
    --alpha 16 \
    --attack-iters 2 \
    --delta-init zero \
    --out-dir "PGD2_NoiseAug_eps32/StepSize16_uniform2_imgNorm" \
    --seed 1 \
    --noise_aug \
    --noise_aug_type uniform \
    --noise_aug_size 2 \
    --image_normalize

python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 32 \
    --alpha 2 \
    --attack-iters 10 \
    --delta-init random \
    --out-dir PGD_baseline \
    --seed 0 \
    --image_normalize


