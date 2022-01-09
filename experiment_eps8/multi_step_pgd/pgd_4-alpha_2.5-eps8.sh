for i in 1 2
do
python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 8 \
    --alpha 2.5 \
    --attack-iters 4 \
    --delta-init random \
    --out-dir PGD_baseline \
    --seed $i \
    --image_normalize
done
