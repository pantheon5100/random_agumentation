NOISE_ADD_POSITION="before after"

for noise_position in $NOISE_ADD_POSITION
do
    for i in 0 1
    do
    python train_pgd_AddFixedNoise.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 16 \
        --attack-iters 1 \
        --delta-init zero \
        --out-dir FGSM_add_fixed_noise \
        --seed $i \
        --image_normalize \
        --add_fixed_noise $noise_position
    done
done

