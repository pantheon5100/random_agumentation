for i in 0 1
do
python train_pgd_NoiseAug_cutout_mixup.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 16 \
    --attack-iters 1 \
    --delta-init random \
    --out-dir Cutout_Mixup \
    --seed $i \
    --image_normalize \
    --cutout \
    --cutout-len 14
done
