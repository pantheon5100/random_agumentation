


for i in 0 1 2 3 4
do
python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 8 \
    --alpha 4 \
    --attack-iters 2 \
    --delta-init zero \
    --out-dir pgd2-baseline-eps8\
    --seed $i \
    --image_normalize
done

