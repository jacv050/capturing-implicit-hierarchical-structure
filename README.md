# [Capturing implicit hierarchical structure in 3D biomedical images with self-supervised hyperbolic representations](https://arxiv.org/pdf/2012.01644.pdf)

![demonstrative figure](images/methods_final_final_final.png)

Code for reproducing the experiments in the paper:
```
@inproceedings{hsu2021capturing,
  title={Capturing implicit hierarchical structure in 3D biomedical images with self-supervised hyperbolic representations},
  author={Hsu, Joy and Gu, Jeffrey and Wu, Gong Her and Chiu, Wah and Yeung, Serena},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

## Prerequisites
`pip install -r -U requirements.txt` or `python3 setup.py install --user`

## Data
The simple synthetic dataset can be downloaded from Google Drive [here](https://drive.google.com/file/d/1mdRuSkXmTof9vq62FSmoZXneUme_97dc/view?usp=sharing) and the irregular synthetic dataset can be downloaded [here](https://drive.google.com/file/d/1XGx8GQlNGCStmxjYatWGBGAW25e2zxTn/view?usp=sharing). Please place the datasets in the `/data` folder. 

## Models

### VAE (`--manifold Euclidean`):
- Prior distribution (`--prior`): `Normal` (`WrappedNormal` is theoretically equivalent)
- Posterior distribution (`--posterior`): `Normal`  (`WrappedNormal` is theoretically equivalent)
- Decoder architecture (`--dec`): `LinearConv` (3D Conv) 
- Encoder architecture (`--enc`): `LinearConv` (3D Conv)
- Triplet loss (`--triplet-loss`): Whether to use hierarchical triplet loss
    - `--triplet-weight`: how much to weight triplet loss relative to the ELBO loss
    
### PVAE (`--manifold PoincareBall`):
- Curvature (`--c`): 1.0
- Prior distribution (`--prior`): `WrappedNormal`
- Posterior distribution (`--posterior`): `WrappedNormal`
- Decoder architecture (`--dec`):
    - `WrappedConv`: 3D Convolutional decoder
    - `GyroConv`: 3D Convolutional decoder with gyroplane convolution as first layer
- Encoder architecture (`--enc`): `WrappedConv` (3D Conv)
- Triplet loss (`--triplet-loss`): Whether to use hierarchical triplet loss
    - `--triplet-weight`: how much to weight triplet loss relative to the (Riemannian) ELBO loss (see Mathieu et al 2019)

## Run experiments

### Synthetic dataset

#### Training

Euclidean:
```
python3 pvae/main.py --model toy_sampled_triplet --manifold Euclidean --latent-dim 2 --hidden-dim 300 --prior Normal --posterior Normal --dec LinearConv --enc LinearConv --lr 5e-5 --epochs 8 --save-freq 1 --batch-size 128 --iwae-samples 5000 --skip-test --name sup_euc_n0 --triplet-loss --triplet-loss-dist --triplet-weight 1e3 --K 8 --save-dir /pasteur/results/jeff-results/capturing-implicit-hierarchical-structure/experiments
```
Hyperbolic:
```
python3 pvae/main.py --model toy_sampled_triplet_conv --manifold PoincareBall --latent-dim 2 --hidden-dim 300 --prior WrappedNormal --posterior WrappedNormal --dec GyroConv --enc WrappedConv --lr 5e-7 --epochs 8 --save-freq 1 --batch-size 128 --iwae-samples 5000 --skip-test --name [your model name] --clip 8 --K 8 --triplet-loss --triplet-loss-dist --triplet-weight 1e3 --save-dir [your save directory]
```

#### Inference

Produces latent representations in preparation for clustering/segmentation. Add the `--eval` flag at the end of the training command, e.g.
```
python3 pvae/main.py --model toy_sampled_triplet_conv --manifold PoincareBall --latent-dim 2 --hidden-dim 300 --prior WrappedNormal --posterior WrappedNormal --dec GyroConv --enc WrappedConv --lr 5e-7 --epochs 8 --save-freq 1 --batch-size 128 --iwae-samples 5000 --skip-test --name [your model name] --clip 8 --K 8 --triplet-loss --triplet-loss-dist --triplet-weight 1e3 --save-dir [your save directory] --eval
```

##### Segmentation
Images are segmentated by clustering the pixelwise latent representations. Some relevant flags are
 - Clustering algorithm (`--cluster-alg`): 
     - `kmeans`: Euclidean K-Means
     - `kmeans_hyp`: Hyperbolic K-Means
 - Clusters (`--clusters`): int, number of clusters
 - Score (`--score`): 
     - `dice`: DICE score
     - `iou`: Intersection over Union
 - Score type (`--score-type`): how detailed ground truth annotations to use
     - `bg`: Foreground/background (corresponds to Level 1 in the paper) (set `--clusters` to `2`)
     - `sc`: Level 2 in the paper (set `--clusters` to `4`)
     - `all`: Level 3 in the paper (set `--clusters` to `8`)

Sample command:
```
python3 pvae/segmentation/segment_and_score.py --model-name test --exp-dir [path to experiment directory] -c 1 --num-images 20 --latent-dim 2 --latent-folder hyperbolic-mus-inference --segmentation-folder hyperbolic-segmented --image-folder hyperbolic-vis --dataset [path to synthetic dataset] --clusters 2 --score dice --hyperbolic --image-height 50 --image-width 50 --image-depth 50 --verbose --score-type bg --cluster-alg kmeans_hyp
```
