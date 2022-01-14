SEED="0 1 2 3 4"

for seed in $SEED; do
  python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 8 \
        --alpha 10 \
        --delta-init random \
        --out-dir "FGSM_NoiseAug_random_eps8_normal_1" \
        --seed $seed \
        --noise_aug \
        --noise_aug_type normal \
        --noise_aug_size 1 \
        --image_normalize
done

for seed in $SEED; do
  python train_fgsm_NoiseAug.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 20 \
        --delta-init random \
        --out-dir "FGSM_NoiseAug_random_eps16_uniform_2" \
        --seed $seed \
        --noise_aug \
        --noise_aug_type uniform \
        --noise_aug_size 2 \
        --image_normalize
done

