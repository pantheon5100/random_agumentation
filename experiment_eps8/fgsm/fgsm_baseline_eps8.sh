# python RA_train_fgsm.py --batch-size 128 \
#         --data-dir /dev/shm \
#         --epochs 30 \
#         --lr-schedule cyclic \
#         --lr-min 0. \
#         --lr-max 0.2 \
#         --weight-decay 5e-4 \
#         --epsilon 8 \
#         --alpha 10 \
#         --delta-init zero \
#         --out-dir fgsm_baseline \
#         --seed 0 \


python RA_train_fgsm.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 8 \
        --alpha 10 \
        --delta-init random \
        --out-dir fgsm_baseline \
        --seed 0 \
        --image_normalize


# python RA_train_fgsm.py --batch-size 128 \
#         --data-dir /dev/shm \
#         --epochs 30 \
#         --lr-schedule cyclic \
#         --lr-min 0. \
#         --lr-max 0.3 \
#         --weight-decay 5e-4 \
#         --epsilon 8 \
#         --alpha 10 \
#         --delta-init random \
#         --out-dir fgsm_baseline \
#         --seed 0
