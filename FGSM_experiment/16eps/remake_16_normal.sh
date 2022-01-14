RA_SIZE="1 2"

for ra_size in $RA_SIZE;do
    python train_fgsm_NoiseAug.py --batch-size 128 \
            --data-dir /dev/shm \
            --epochs 30 \
            --lr-schedule cyclic \
            --lr-min 0. \
            --lr-max 0.3 \
            --weight-decay 5e-4 \
            --epsilon 16 \
            --alpha 20 \
            --delta-init zero \
            --out-dir "remake_noise_align_normal_eps16_/size_"$ra_size \
            --seed 5 \
            --noise_aug \
            --noise_aug_type normal \
            --noise_aug_size $ra_size \
            --image_normalize
done



