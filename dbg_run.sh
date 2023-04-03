#!/bin/bash

python3 -m pdb pvae/main.py --model behaviour_sampled_triplet_conv --manifold PoincareBall --latent-dim 2 --hidden-dim 300 --prior WrappedNormal --posterior WrappedNormal --dec GyroConv --enc WrappedConv --lr 5e-7 --epochs 100 --save-freq 1 --batch-size 128 --iwae-samples 5000 --skip-test --clip 8 --K 8 --triplet-loss --triplet-loss-dist --triplet-weight 1e3 --save-dir results
