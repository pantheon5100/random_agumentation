for i in 1 2 
do
    python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 40 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 8 \
    --attack-iters 2 \
    --delta-init zero \
    --out-dir PGD_baseline \
    --seed $i \
    --noise_aug \
    --noise_aug_type uniform \
    --noise_aug_size 2 \
    --zero_one_clamp 0 \
    --image_normalize
done

