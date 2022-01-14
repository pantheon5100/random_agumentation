


for i in 0 1 2 3 4
do
python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 8 \
    --alpha 1.6 \
    --attack-iters 10 \
    --delta-init zero \
    --out-dir pgd10-noise-aug \
    --seed $i \
    --noise_aug \
    --noise_aug_type normal \
    --noise_aug_size 1 \
    --image_normalize
done

