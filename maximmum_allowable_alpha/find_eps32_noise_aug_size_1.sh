
ALPHA="8 32 40 56 72 88 120 152"

for alpha in $ALPHA; do
python train_fgsm_NoiseAug_max_allowable.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 32 \
        --alpha $alpha \
        --delta-init zero \
        --out-dir "Maximum/eps32_alpha"$alpha"_noise_aug_size1" \
        --seed 1 \
        --noise_aug \
        --noise_aug_type uniform \
        --noise_aug_size 1 \
        --image_normalize
done


