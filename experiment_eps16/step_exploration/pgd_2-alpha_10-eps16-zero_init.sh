
for alpha_size in 10 12 14 16
do
    for i in 1 2
    do
        python train_pgd_NoiseAug.py --batch-size 128 \
            --data-dir /dev/shm \
            --epochs 30 \
            --lr-schedule cyclic \
            --lr-min 0. \
            --lr-max 0.3 \
            --weight-decay 5e-4 \
            --epsilon 16 \
            --alpha $alpha_size \
            --attack-iters 2 \
            --delta-init zero \
            --out-dir PGD_baseline \
            --seed $i \
            --image_normalize
    done
done


# 