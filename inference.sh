#!/bin/bash

python3 pvae/main.py --model behaviour_sampled_triplet_conv --manifold PoincareBall \
	--latent-dim 10 --hidden-dim 300 --prior WrappedNormal --posterior WrappedNormal \
	--dec GyroConv --enc WrappedConv --lr 5e-7 --epochs 8 --save-freq 1 --batch-size 128 \
	--iwae-samples 5000 --skip-test \
	--name plancha-test --clip 8 --K 8 --triplet-loss \
	--triplet-loss-dist --triplet-weight 1e3 \
	--save-dir exp-dir/hyperbolic-mus-inference --eval
