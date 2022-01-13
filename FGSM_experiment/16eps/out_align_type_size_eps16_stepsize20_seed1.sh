ALIGN_TETHOD="KL_AE_NAE NAE2GT"
ALIGH_NOISE_SIZE="1 2"

sleep 5

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for align_method in $ALIGN_TETHOD; do
    for align_noise_size in $ALIGH_NOISE_SIZE;do
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
                --out-dir "FGSM_out_algin_size-eps16-step_size20_/size_"$align_noise_size \
                --seed 1 \
                --out_align_method $align_method \
                --out_align_noise $align_noise_size \
                --image_normalize
    done
done



