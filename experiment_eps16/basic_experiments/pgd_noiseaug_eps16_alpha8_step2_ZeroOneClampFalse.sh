# ###### 30 epochs and max lr 0.3
# python train_pgd_NoiseAug.py --batch-size 128 \
#     --data-dir /dev/shm \
#     --epochs 30 \
#     --lr-schedule cyclic \
#     --lr-min 0. \
#     --lr-max 0.3 \
#     --weight-decay 5e-4 \
#     --epsilon 16 \
#     --alpha 8 \
#     --attack-iters 2 \
#     --delta-init zero \
#     --out-dir PGD_baseline \
#     --seed 0 \
#     --noise_aug \
#     --noise_aug_type uniform \
#     --noise_aug_size 2 \
#     --zero_one_clamp 0 \
#     --image_normalize


###### 40 epochs and max lr 0.2
# python train_pgd_NoiseAug.py --batch-size 128 \
#     --data-dir /dev/shm \
#     --epochs 40 \
#     --lr-schedule cyclic \
#     --lr-min 0. \
#     --lr-max 0.2 \
#     --weight-decay 5e-4 \
#     --epsilon 16 \
#     --alpha 8 \
#     --attack-iters 2 \
#     --delta-init zero \
#     --out-dir PGD_baseline \
#     --seed 0 \
#     --noise_aug \
#     --noise_aug_type uniform \
#     --noise_aug_size 2 \
#     --zero_one_clamp 0 \
#     --image_normalize

###### 30 epochs and max lr 0.2
# python train_pgd_NoiseAug.py --batch-size 128 \
#     --data-dir /dev/shm \
#     --epochs 30 \
#     --lr-schedule cyclic \
#     --lr-min 0. \
#     --lr-max 0.2 \
#     --weight-decay 5e-4 \
#     --epsilon 16 \
#     --alpha 8 \
#     --attack-iters 2 \
#     --delta-init zero \
#     --out-dir PGD_baseline \
#     --seed 0 \
#     --noise_aug \
#     --noise_aug_type uniform \
#     --noise_aug_size 2 \
#     --zero_one_clamp 0 \
#     --image_normalize


##### 40 epochs and max lr 0.3
# python train_pgd_NoiseAug.py --batch-size 128 \
#     --data-dir /dev/shm \
#     --epochs 40 \
#     --lr-schedule cyclic \
#     --lr-min 0. \
#     --lr-max 0.3 \
#     --weight-decay 5e-4 \
#     --epsilon 16 \
#     --alpha 8 \
#     --attack-iters 2 \
#     --delta-init zero \
#     --out-dir PGD_baseline \
#     --seed 0 \
#     --noise_aug \
#     --noise_aug_type uniform \
#     --noise_aug_size 2 \
#     --zero_one_clamp 0 \
#     --image_normalize


###### 40 epochs and max lr 0.2 + zero_one_clamp 1
python train_pgd_NoiseAug.py --batch-size 128 \
    --data-dir /dev/shm \
    --epochs 40 \
    --lr-schedule cyclic \
    --lr-min 0. \
    --lr-max 0.2 \
    --weight-decay 5e-4 \
    --epsilon 16 \
    --alpha 8 \
    --attack-iters 2 \
    --delta-init zero \
    --out-dir PGD_baseline \
    --seed 0 \
    --noise_aug \
    --noise_aug_type uniform \
    --noise_aug_size 2 \
    --zero_one_clamp 1 \
    --image_normalize