
NOISEAUG_POSITION="before after"
for noiseaug_position in $NOISEAUG_POSITION
do
for i in 0 1
do
python train_pgd_NoiseAug_AblationStudy_NoiseAugAddBeforeOrAfter.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 8 \
    --attack-iters 2 \
    --delta-init zero \
    --out-dir NoiseAugAddBeforeOrAfter \
    --seed $i \
    --noise_aug \
    --noise_aug_type normal \
    --noise_aug_size 2 \
    --image_normalize \
    --noise_aug_position $noiseaug_position
done
done



NOISEAUG_POSITION="before after"
for noiseaug_position in $NOISEAUG_POSITION
do
for i in 0 1
do
python train_pgd_NoiseAug_AblationStudy_NoiseAugAddBeforeOrAfter.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.3 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 8 \
    --attack-iters 1 \
    --delta-init zero \
    --out-dir NoiseAugAddBeforeOrAfter \
    --seed $i \
    --noise_aug \
    --noise_aug_type normal \
    --noise_aug_size 2 \
    --image_normalize \
    --noise_aug_position $noiseaug_position
done
done
