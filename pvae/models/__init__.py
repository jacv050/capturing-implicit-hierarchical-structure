from .tabular import Tree as VAE_tree, CSV as VAE_csv
from .mnist import Mnist as VAE_mnist
from .toy_sampled_triplet import ToySampledTriplet as VAE_toy_sampled_triplet
from .toy_sampled_triplet_conv import ToySampledTripletConv as VAE_toy_sampled_triplet_conv
from .behaviour_sampled_triplet_conv import BehaviourSampledTripletConv as VAE_behaviour_sampled_triplet_conv
from .behaviour_video_triplet_conv import BehaviourVideoTripletConv as VAE_behaviour_video_triplet_conv


__all__ = [VAE_csv, VAE_tree, VAE_mnist, VAE_toy_sampled_triplet, VAE_toy_sampled_triplet_conv, VAE_behaviour_sampled_triplet_conv]
