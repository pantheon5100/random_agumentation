ALIGN_TETHOD="KL_AE_NAE NAE2GT"
ALIGH_NOISE_SIZE="1 2"


#Experiment of [out_align_method,align_noise]  for [eps= 8 ,norm]
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
                --out-dir "3rd_16/align_"$align_method"_"$i"_norm" \
                --seed 2 \
                --out_align_method $align_method \
                --out_align_noise $i \
                --image_normalize \
                --comment fj_0111_10am
    done
done

#Experiment of [out_align_method,align_noise]  for [eps= 8 ,non-norm]
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
                --out-dir "3rd_16/align_"$align_method"_"$i"_non-norm" \
                --seed 2 \
                --out_align_method $align_method \
                --out_align_noise $i \
                --comment fj_0110_10am
    done
done