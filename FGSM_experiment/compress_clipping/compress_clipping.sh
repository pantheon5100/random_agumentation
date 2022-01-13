

# FACTOR="7 8"
# ALPHA="10 11"

# #Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
# for factor in $FACTOR; do
# for alpha in $ALPHA; do
#     for i in 0 1;do
#         python train_fgsm_NoiseAug_compress_clipping.py --batch-size 128 \
#                 --data-dir /dev/shm \
#                 --epochs 30 \
#                 --lr-schedule cyclic \
#                 --lr-min 0. \
#                 --lr-max 0.3 \
#                 --weight-decay 5e-4 \
#                 --epsilon 8 \
#                 --alpha $alpha \
#                 --delta-init zero \
#                 --out-dir "Compress_clipping-eps8" \
#                 --seed $i \
#                 --clamp_compress_factor $factor \
#                 --image_normalize
#     done
# done
# done


FACTOR="18 16 14 12 10"
# ALPHA="10 11"

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for factor in $FACTOR; do
    for i in 0 1;do
        python train_fgsm_NoiseAug_compress_clipping.py --batch-size 128 \
                --data-dir /dev/shm \
                --epochs 30 \
                --lr-schedule cyclic \
                --lr-min 0. \
                --lr-max 0.3 \
                --weight-decay 5e-4 \
                --epsilon 8 \
                --alpha 10 \
                --delta-init random \
                --out-dir "Compress_clipping-eps8-delat_RS" \
                --seed $i \
                --clamp_compress_factor $factor \
                --image_normalize
    done
done

