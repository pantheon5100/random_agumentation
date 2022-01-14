NOISE_FACTOR="0.25 0.5 0.75"

sleep 5

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for noise_factor in $NOISE_FACTOR;do
    python train_fgsm_NoiseAug.py --batch-size 128 \
            --data-dir /dev/shm \
            --epochs 30 \
            --lr-schedule cyclic \
            --lr-min 0. \
            --lr-max 0.3 \
            --weight-decay 5e-4 \
            --epsilon 8 \
            --alpha 10 \
            --delta-init zero \
            --out-dir "NAE2GT_factor_eps8/factor_"$noise_factor \
            --seed 0 \
            --out_align_method NAE2GT \
            --out_align_noise 2 \
            --nae_to_gt_factor $noise_factor \
            --image_normalize
done



