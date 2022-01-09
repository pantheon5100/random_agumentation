ALIGN_TETHOD="KL_AE_NAE NAE2GT"
ALIGH_NOISE_SIZE="1 2"

# KL,NS 1,Norm  && NAE2GT,NS 1,Norm
for align_method in $ALIGN_TETHOD; do
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
            --out_align_noise 1 \
            --image_normalize \
            --comment fj_0109_21pm
done

#Experiment of [out_align_method,align_noise]  for [eps=16 ,non-norm]
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
                --comment fj_0109_21pm
    done
done

#Experiment of [out_align_method,align_noise]  for [eps=8 ,non-norm]
# for align_method in $ALIGN_TETHOD; do
#     for i in $ALIGH_NOISE_SIZE;do
#         python RA_train_fgsm.py --batch-size 128 \
#                 --data-dir /dev/shm \
#                 --epochs 30 \
#                 --lr-schedule cyclic \
#                 --lr-min 0. \
#                 --lr-max 0.3 \
#                 --weight-decay 5e-4 \
#                 --epsilon 16 \
#                 --alpha 20 \
#                 --delta-init random \
#                 --out-dir 2_align_method_comparision \
#                 --seed 0 \
#                 --out_align_method $align_method \
#                 --out_align_noise $i \
#                 --comment fj_0109_21pm
#     done
# done