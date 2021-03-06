ALIGN_TETHOD="KL_AE_NAE NAE2GT"
ALIGH_NOISE_SIZE="1 2"


for align_method in $ALIGN_TETHOD; do
    for i in $ALIGH_NOISE_SIZE;do
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
                --out-dir 2_align_method_comparision \
                --seed 0 \
                --out_align_method $align_method \
                --out_align_noise $i \
                --image_normalize \
                --comment fj_0109_21pm
    done
done



# RA_TYPE="uniform normal"

# for ra_type in $RA_TYPE; do
#     python RA_train_fgsm.py --batch-size 128 \
#             --data-dir /dev/shm \
#             --epochs 30 \
#             --lr-schedule cyclic \
#             --lr-min 0. \
#             --lr-max 0.3 \
#             --weight-decay 5e-4 \
#             --epsilon 16 \
#             --alpha 20 \
#             --delta-init random \
#             --out-dir random_augmentation \
#             --seed 0 \
#             --random_augmentation \
#             --ra_type $ra_type \
#             --ra_size 2 \
#             --image_normalize
# done


# for ra_type in $RA_TYPE; do
#     python RA_train_fgsm.py --batch-size 128 \
#             --data-dir /dev/shm \
#             --epochs 30 \
#             --lr-schedule cyclic \
#             --lr-min 0. \
#             --lr-max 0.3 \
#             --weight-decay 5e-4 \
#             --epsilon 16 \
#             --alpha 20 \
#             --delta-init random \
#             --out-dir random_augmentation \
#             --seed 0 \
#             --random_augmentation \
#             --ra_type $ra_type \
#             --ra_size 2 \
# done
