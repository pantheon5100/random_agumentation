step1=8
step2=8
for i in 1 2
do
    python train_pgd_NoiseAug_StepSize.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
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
    --image_normalize \
    --noise_aug_size_step1 $step1 \
    --noise_aug_size_step2 $step2
done



step1=10
step2=10
for i in 1 2
do
    python train_pgd_NoiseAug_StepSize.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
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
    --image_normalize \
    --noise_aug_size_step1 $step1 \
    --noise_aug_size_step2 $step2
done



step1=12
step2=4
for i in 1 2
do
    python train_pgd_NoiseAug_StepSize.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
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
    --image_normalize \
    --noise_aug_size_step1 $step1 \
    --noise_aug_size_step2 $step2
done



step1=16
step2=4
for i in 1 2
do
    python train_pgd_NoiseAug_StepSize.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
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
    --image_normalize \
    --noise_aug_size_step1 $step1 \
    --noise_aug_size_step2 $step2
done




step1=12
step2=12
for i in 1 2
do
    python train_pgd_NoiseAug_StepSize.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 30 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
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
    --image_normalize \
    --noise_aug_size_step1 $step1 \
    --noise_aug_size_step2 $step2
done




