#!/bin/bash

python3 pvae/main.py --model behaviour_video_triplet_conv \
	--manifold PoincareBall \
	--latent-dim 2 --hidden-dim 600 \
	--prior Normal --posterior Normal \
	--dec LinearVideoConv --enc LinearConv --lr 5e-10 \
	--epochs 8 --save-freq 1 --batch-size 128 \
	--iwae-samples 5000 --skip-test \
	--name plancha-test --clip 8 --K 8 --triplet-loss \
	--triplet-loss-dist --triplet-weight 1e2 \
	--save-dir results_euclidean --eval
