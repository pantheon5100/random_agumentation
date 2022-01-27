
python train_fgsm_NoiseAug_supervised_linearity_measure.py --batch-size 128 \
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
    --out-dir supervised_baseline_eps8 \
    --seed 0 \
    --zero_one_clamp 0 \
    --cutout \
    --n_holes 1 \
    --length 16 \
    --image_normalize
