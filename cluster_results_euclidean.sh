#!/bin/bash

python3 pvae/segmentation/segment_and_score_video_dc.py --model-name plancha-test \
    --exp-dir results_euclidean -c 1 \
    --latent-dim 2 \
    --latent-folder hyperbolic-mus-inference \
    --segmentation-folder hyperbolic-segmented \
    --image-folder hyperbolic-vis --dataset p1/validation \
    --clusters 5 --score recall --image-height 50 \
    --image-width 50 --image-depth 50 --verbose --score-type all \
    --groundtruth-folder /workspace/dataset \
    --visualize --processes 1\
    --cluster-alg kmeans

#--num-images 20
