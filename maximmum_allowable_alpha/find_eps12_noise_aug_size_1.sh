
ALPHA="6 8 12 16 32 40 56 72 88 120 152"

for alpha in $ALPHA; do
python train_fgsm_NoiseAug_max_allowable.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 12 \
        --alpha $alpha \
        --delta-init zero \
        --out-dir "Maximum/eps12_alpha"$alpha \
        --seed 1 \
        --noise_aug \
        --noise_aug_type uniform \
        --noise_aug_size 1 \
        --image_normalize
done


