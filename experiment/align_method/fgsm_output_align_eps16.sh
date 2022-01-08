ALIGN_TETHOD="CS_AE_NAE KL_AE_NAE NAE2GT"

for align_method in $ALIGN_TETHOD; do
    python RA_train_fgsm.py --batch-size 128 \
            --data-dir /dev/shm \
            --epochs 30 \
            --lr-schedule cyclic \
            --lr-min 0. \
            --lr-max 0.3 \
            --weight-decay 5e-4 \
            --epsilon 16 \
            --alpha 10 \
            --delta-init random \
            --out-dir align_method_comparision \
            --seed 0 \
            --out_align_method $align_method \
            --out_align_noise 2 \
            --image_normalize
done
