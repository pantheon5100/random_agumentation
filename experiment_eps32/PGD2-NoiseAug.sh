NORMAL_SIZE="1 2"
SEED="0 1 2 3 4"

sleep 5

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for normal_size in $NORMAL_SIZE; do
    for s in $SEED;do
        # eps32 + alpha32 + uniform*2
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
            --out-dir "PGD2_NoiseAug_eps32/StepSize16_normal"$normal_size"_imgNorm" \
            --seed $s \
            --noise_aug \
            --noise_aug_type normal \
            --noise_aug_size  $normal_size\
            --image_normalize

    done
done


