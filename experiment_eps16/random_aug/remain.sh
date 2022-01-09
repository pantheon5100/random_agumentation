

python RA_train_fgsm.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 20 \
        --delta-init random \
        --out-dir align_method_comparision_1 \
        --seed 0 \
        --out_align_method KL_AE_NAE \
        --out_align_noise 1 \
        --comment fj_0109_15pm

python RA_train_fgsm.py --batch-size 128 \
        --data-dir /dev/shm \
        --epochs 30 \
        --lr-schedule cyclic \
        --lr-min 0. \
        --lr-max 0.3 \
        --weight-decay 5e-4 \
        --epsilon 16 \
        --alpha 20 \
        --delta-init random \
        --out-dir align_method_comparision_1 \
        --seed 0 \
        --out_align_method NAE2GT \
        --out_align_noise 1 \
        --comment fj_0109_15pm

