#!/bin/bash

python3 pvae/main.py --model behaviour_video_triplet_conv \
	--manifold PoincareBall \
	--latent-dim 2 --hidden-dim 400 \
	--prior WrappedNormal --posterior WrappedNormal \
	--dec GyroVideoConv --enc WrappedVideoConv --lr 5e-10 \
	--epochs 8 --save-freq 1 --batch-size 128 \
	--iwae-samples 5000 --skip-test \
	--name plancha-test --np_zeropadding --clip 8 --K 8 --triplet-loss \
	--triplet-loss-dist --triplet-weight 1e3 \
	--save-dir exp-dir-post-robot2022 --eval
