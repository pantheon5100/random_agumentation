RA_TYPE="uniform normal"
RA_SIZE="1 2"

#Experiment of [ra_type,ra_size]  for [eps= 8 ,norm]
for ra_type in $RA_TYPE; do
    for i in $RA_SIZE;do
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
                --out-dir "2nd_8/ra_"$ra_type"_"$i"_norm" \
                --seed 1 \
                --random_augmentation \
                --ra_type $ra_type \
                --ra_size $i \
                --image_normalize \
                --comment fj_0110_23pm
    done
done

#Experiment of [ra_type,ra_size]  for [eps= 8 ,non-norm]
for ra_type in $RA_TYPE; do
    for i in $RA_SIZE;do
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
                --out-dir "2nd_8/ra_"$ra_type"_"$i"_norm" \
                --seed 1 \
                --random_augmentation \
                --ra_type $ra_type \
                --ra_size $i \
                --comment fj_0110_23pm
    done
done
