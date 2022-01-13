RA_TYPE="uniform normal"
RA_SIZE="1 2 3"

sleep 5

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for ra_type in $RA_TYPE; do
    for ra_size in $RA_SIZE;do
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
                --out-dir "FGSM_NoiseAug_type_size-eps8-step_size10" \
                --seed 1 \
                --noise_aug \
                --noise_aug_type $ra_type \
                --noise_aug_size $ra_size \
                --image_normalize
    done
done



