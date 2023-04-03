#!/bin/bash

python3 pvae/main.py --model behaviour_video_triplet_conv \
    --manifold PoincareBall \
    --latent-dim 2 --hidden-dim 300 \
    --prior WrappedNormal --posterior WrappedNormal \
    --dec GyroVideoConv --enc WrappedVideoConv \
    --lr 1e-7 --epochs 10000 --save-freq 1 --batch-size 128 \
    --iwae-samples 5000 --clip 8 --K 8 \
    --triplet-loss --triplet-loss-dist --seed 4180631546 \
    --no_padding --np_zeropadding --triplet-weight 1e7 --save-dir prueba_borrar
    #--skip-test
#--lr 5e-5 5e-7

#robot2022
#python3 pvae/main.py --model behaviour_video_triplet_conv \
    #--manifold PoincareBall \
    #--latent-dim 2 --hidden-dim 600 \
    #--prior WrappedNormal --posterior WrappedNormal \
    #--dec GyroVideoConv --enc WrappedConv \
    #--lr 5e-10 --epochs 700 --save-freq 1 --batch-size 1 \
    #--iwae-samples 5000 --skip-test --clip 8 --K 8 \
    #--triplet-loss --triplet-loss-dist \
    #--no_padding --triplet-weight 1e2 --save-dir prueba_borrar
    #TODO adaptar GyroConv -> en vez GyroVideoConv
