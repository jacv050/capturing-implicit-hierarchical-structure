import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.optimize import linear_sum_assignment
from skimage.transform import rescale
from sklearn.cluster import KMeans
from clustering import PoincareKMeansParallel
from itertools import islice
import argparse
import os
import pickle
import torch
import time
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model-name', type=str, default='Model Name', help='Name of trained model')
parser.add_argument('--exp-dir', type=str, default='', help='Path to experiment directory')
parser.add_argument('-c', type=float, default=1., help='(Negative) curvature')
parser.add_argument('--num-images', type=int, default=None, help='How many images representations are in mus.p')
parser.add_argument('--latent-dim', type=int, default=2, help='Latent dimension of our embedding space')
parser.add_argument('--latent-folder', type=str, default='hyperbolic-mus-inference', help='File (mus.p) where the image representations produced by inference are stored')
parser.add_argument('--segmentation-folder', type=str, default='hyperbolic-segmented', help='Folder where segmentations are stored')
parser.add_argument('--image-folder', type=str, default='hyperbolic-vis', help='Folder where visualizations are stored')
parser.add_argument('--groundtruth-folder', type=str, default='/pasteur/data/hierarchical-toy-dataset', help='Folder where the groundtruth segmentations are stored')
parser.add_argument('--dataset', type=str, default='toy_v2/test', help='Name of the dataset to evaluate on')
parser.add_argument('--hyperbolic', action='store_true', default=False, help='Whether we are using hyperbolic latent space')
parser.add_argument('--clusters', type=int, default=2, help='Number of clusters')
parser.add_argument('--score', type=str, default='', help='Scoring function for segmentation')
parser.add_argument('--image-width', type=int, default=46, help='Width (dim 0 size) of input images')
parser.add_argument('--image-height', type=int, default=46, help='Height (dim 1 size) of input images')
parser.add_argument('--image-depth', type=int, default=46, help='Depth (dim 2 size) of input images')
parser.add_argument('--verbose', action='store_true', default=False, help='If true, print more messages')
parser.add_argument('--bg', action='store_true', default=False, help='If true, do foreground/background segmentations')
parser.add_argument('--visualize', action='store_true', default=False, help='If true, visualize segmentations')
parser.add_argument('--cluster-alg', type=str, default='kmeans', choices=['kmeans', 'kmeans_hyp'], help='Clustering algorithm for producing segmentations')
parser.add_argument('--score-type', type=str, default='bg', help='Which level (1, 2, 3) we are evaluating on')
parser.add_argument('--nearest-neighbors', type=int, default=10, help='Number of nearest neighbors for algorithms that use kNN')
parser.add_argument('--downsample', type=float, default=0, help='Downsample images for quicker evaluation')
parser.add_argument('--processes', type=int, default=1, help='Num processes for algorithms that have some parallelization')
parser.add_argument('--cluster_merged', action='store_true', default=False, help='If true, calculate cluster merging clips')
#parser.add_argument('--likeframes', type=bool, default=True, help='Cluster like frames in dimensions X and Y. Z is considered frames.')
args = parser.parse_args()

# Create folders
mus_path = os.path.join(args.exp_dir, args.latent_folder, args.model_name, 'mus.p')
image_path = os.path.join(args.exp_dir, args.image_folder, args.model_name)
if not os.path.exists(image_path):
    os.makedirs(image_path)
save_path = os.path.join(args.exp_dir, args.segmentation_folder, args.model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# Utils
def dice(points1, points2):
    intersections = set(points1).intersection(set(points2))
    union = set(points1).union(set(points2))
    return (2 * len(intersections))/(len(intersections) + len(union))

def accuracy(points1, points2):
    intersections = set(points1).intersection(set(points2))
    #union = set(points1).union(set(points2))
    return len(intersections)/len(points2)

def iou(points1, points2):
    intersections = set(points1).intersection(set(points2))
    union = set(points1).union(set(points2))
    return len(intersections) / len(union)

def plot_test(image):
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.savefig(os.path.join(image_path, 'test_image.png'))

def get_score_video(x, i, y, j, score='dice'):
    """
    x, y have same shape
    """
    x_points = []
    y_points = []
    
    for w in range(x.shape[0]):
        if x[w] == i:
            x_points.append((w))
        if y[w] == j:
            y_points.append((w))
    
    if score == 'iou':
        return iou(x_points, y_points)
    if score =='dice':
        return dice(x_points, y_points) 
    if score == 'accuracy':
        return accuracy(x_points, y_points)

def get_score(x, i, y, j, score='dice'):
    """
    x, y have same shape
    """
    x_points = []
    y_points = []
    for w in range(x.shape[0]):
        for l in range(x.shape[1]):
            for h in range(x.shape[2]):
                if x[w][l][h] == i:
                    x_points.append((w, l, h))
                if y[w][l][h] == j:
                    y_points.append((w, l, h))
    if score == 'iou':
        return iou(x_points, y_points)
    if score =='dice':
        return dice(x_points, y_points) 
    
def get_best_scores_videos(seg, truth, score='dice', verbose=False, bg=False):
    if seg.shape != truth.shape:#TODO check
        reshaped_truth = rescale(truth, seg.shape[0]/truth.shape[0], anti_aliasing=False)
        reshaped_truth = np.rint(reshaped_truth)
    else:
        reshaped_truth = truth
    #if bg:
    #    reshaped_truth = np.where(reshaped_truth > 0.5, 1, 0)
    
    uniq = np.unique(reshaped_truth)
    
    cost_matrix = np.zeros((len(np.unique(seg)), len(uniq)))
    if verbose:
        print(cost_matrix.shape)
    
    for i in range(len(np.unique(seg))):
        for j in range(len(uniq)):
            cost_matrix[i][j] = get_score_video(seg, i, reshaped_truth, uniq[j], score=score)
    cost_matrix *= -1
    if verbose:
        print(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_ious = cost_matrix[row_ind, col_ind].sum()
    if verbose:
        print(total_ious)
        print(len(row_ind))
        print((total_ious / len(row_ind)) * -1)
    return (total_ious / len(row_ind)) * -1

def cluster(all_latent, num_images):
    im_clusters = []
    im_centroids = []
    #for i in range(num_images):
    for i in range(len(all_latent)):
        im_latent = all_latent[i][:]
        print(len(all_latent[i][0]))
        if args.downsample > 0:
            new_latent = []
            im_latent = np.reshape(im_latent, (args.image_width, args.image_height, args.image_depth, args.latent_dim))
            for d in range(args.latent_dim):
                rescaled_im = rescale(im_latent[:, :, :, d], 1/args.downsample, anti_aliasing=False)
                new_latent.append(rescaled_im)
            im_latent = np.moveaxis(np.array(new_latent), 0, -1)
        im_latent = np.reshape(im_latent, (-1, args.latent_dim))
        if args.cluster_alg == 'kmeans':
            model = KMeans(n_clusters=args.clusters,
                           random_state=0,
                           n_init=10,
                           max_iter=100,#100
                           tol=1e-10,
                           verbose=args.verbose)
            model.fit(im_latent)
            im_clusters.append(model.labels_)
            im_centroids.append(model.cluster_centers_)
        elif args.cluster_alg == 'kmeans_hyp' and args.hyperbolic == True:
            model = PoincareKMeansParallel(
                n_dim=args.latent_dim, 
                n_clusters=args.clusters, 
                n_init=args.processes, 
                max_iter=100, 
                tol=1e-10, 
                verbose=args.verbose,
                processes=args.processes,
            )
            model.fit(im_latent)
            im_clusters.append(model.labels_)
            im_centroids.append(model.cluster_centers_)
    return im_clusters, im_centroids

#samples_clip = samples per clip
def score_videos(im_clusters, lsize_of_video):
    scores = []
    #len(lsize_of_video) nos informa de la cantidad de clips
    #lsize_of_video contiene la cantidad de clips para cada video
    #len(lsize_of_video) debe ser igual a len(im_clusters)
    if(len(lsize_of_video) != len(im_clusters)):
        print("Check this... Why I'm in... I shouldn't")
        print(im_clusters)
        print("{} {}".format(len(lsize_of_video), len(im_clusters)))
        sys.exit()
    for i in range(len(lsize_of_video)):
        truth = None#TODO Cargar ground truth
        #truth = pickle.load(open(os.path.join(args.groundtruth_folder, args.dataset, ('truth_' + args.score_type + '_{}.pt').format(i)), 'rb'))
        truth = torch.load(os.path.join(args.groundtruth_folder, args.dataset, ('truth_' + args.score_type + '_{}.pt').format(i)))
        truth = np.reshape(truth, lsize_of_video[i])
        cluster = np.reshape(im_clusters[i], lsize_of_video[i])
        s = get_best_scores_videos(cluster, truth, args.score, args.verbose, args.bg)
        scores.append(s)
    print('{} scores for all images:'.format(args.score), scores)
    print('Average {} score:'.format(args.score), np.mean(scores))
    return scores

def score(im_clusters):
    scores = []
    for i in range(len(im_clusters)):
        truth = pickle.load(open(os.path.join(args.groundtruth_folder, args.dataset, ('truth_' + args.score_type + '_{}.p').format(i)), 'rb'))
        if args.downsample > 0:
            cluster = np.reshape(im_clusters[i], (int(np.rint(args.image_width/args.downsample)), int(np.rint(args.image_height/args.downsample)), int(np.rint(args.image_depth/args.downsample))))
        else:
            cluster = np.reshape(im_clusters[i], (args.image_width, args.image_height, args.image_depth))
        s = get_best_scores_videos(cluster, truth, args.score, args.verbose, args.bg)
        scores.append(s)
    print('{} scores for all images:'.format(args.score), scores)
    print('Average {} score:'.format(args.score), np.mean(scores))
    return scores

def visualize_mus(r_latent):
    '''
    Currently only works for latent dim 2
    '''
    for i in range(args.num_images):
        idxs = np.random.choice(range(0, all_latent.shape[0]), 500)
    
        # plot mus
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        plt.scatter(all_latent[idxs, 0], all_latent[idxs, 1], c='blue')
    
        # plot circle
        circle_x = np.array([np.cos(theta * np.pi/180) for theta in range(0, 360, 1)])
        circle_y = np.array([np.sin(theta * np.pi/180) for theta in range(0, 360, 1)])
        f = 1/np.sqrt(args.c)
        circle_x *= f
        circle_y *= f
        plt.scatter(circle_x, circle_y, c='black')
    
        # plot origin
        plt.scatter(0, 0, c='black')
        
        plt.savefig(os.path.join(image_path, 'mus_{}.png'.format(i)))
        
def visualize_clusters(r_latent, im_clusters, im_centroids, num_images):
    '''
    Currently only works for latent dim 2
    '''
    colormap = get_cmap('jet')
    
    for j in range(num_images):
        all_latent = r_latent[j][:, :]
        # sample 10000 points for speed
        idxs = np.random.choice(range(0, all_latent.shape[0]), 10000)
        all_latent = all_latent[idxs, :]
        
        clust_labels = im_clusters[j]
        centroids = im_centroids[j]
        colors = [colormap(x/args.clusters) for x in clust_labels]
        colors = np.array(colors)[idxs]
        
        # plot
        plt.figure(figsize=(8,8))
        plt.scatter(all_latent[:,0],all_latent[:,1],alpha=0.3,c=colors)
        plt.xlim([np.min(all_latent[:,0]), np.max(all_latent[:,0])])
        plt.ylim([np.min(all_latent[:,1]), np.max(all_latent[:,1])])
        for i in range(args.clusters):
            plt.scatter(centroids[i,0],centroids[i,1],alpha=1.0,c="red")
        plt.savefig(os.path.join(image_path, 'cluster_{}.png'.format(j)))    
    
if __name__ == '__main__':
    t0 = time.time()
    folder = 'patched'
    clips_index_path = 'preprocessed/{}/clips_test_index.py'.format(folder)
    print('Loading mus...')
    all_latent = pickle.load(open(mus_path, 'rb'))
    clips_index = torch.load(clips_index_path)
    
    #Prepare video lengths
    dim = clips_index["dim"]
    sx, sy, sz, shapes = dim
    clips_per_video = [len(v) for k, v in clips_index["videos"].items()]
    
    #Listing latent spaces per video
    r_latent = []
    index = 0
    #for s in clips_per_video:
    for i in range(len(clips_per_video)):
        s = clips_per_video[i]
        r_latent.append(all_latent[index:index+s])
        index += s

    #num_images = args.num_images = num_clips
    if(args.num_images is None):
        num_images = len(clips_per_video)
    else:
        num_images = args.num_images

    r_latent_cluster = r_latent
    if args.cluster_merged:
        r_latent_cluster = [np.concatenate(r_latent, axis=0)]

    print('Clustering...')
    clusters, centroids = cluster(r_latent_cluster, num_images)
    
    if(args.cluster_merged):
        print(clusters[0].shape)
        l = iter(clusters[0])
        clusters = [list(islice(l, i)) for i in clips_per_video]
        #clusters = np.repeat(np.expand_dims(clusters, axis=0), num_images, axis=0)
        centroids = np.repeat(np.expand_dims(np.array(centroids[0]), axis=0), num_images, axis=0)
    
    if args.visualize:
        print('Visualizing clusters...')
        visualize_clusters(r_latent, clusters, centroids, num_images)
    if args.score != '':
        print('Scoring segmentations...')
        #_ = score(clusters)
        _ = score_videos(clusters, clips_per_video)
    pickle.dump(clusters, open(os.path.join(save_path, 'segmented{}.p'.format(args.clusters)), 'wb'))
    print('Elapsed time: ', time.time() - t0)
