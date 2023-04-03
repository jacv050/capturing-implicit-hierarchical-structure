import numpy as np
from random import sample
import multiprocessing
import multiprocessing.dummy as mp
from itertools import cycle
from tqdm import tqdm
import sys
from time import time

def clamp(a, eps=1e-12):
    """
    Makes things non-negative
    """
    return np.maximum((a - eps), 0) + eps

def arcosh(x):
    c = clamp(x**2 - 1)
    return np.log(clamp(x + np.sqrt(c)))

def g(x):
    return (2 * arcosh(1 + 2 * x))/clamp(np.sqrt(x**2 + x))

def frechet_mean(x, iterations=1):
    yk = x[0]
    for k in range(iterations):
        a = 0
        b = 0
        c = 0
        for l in range(len(x)):
            num  = np.inner(x[l] - yk, x[l] - yk)
            denom1 = 1 - np.inner(x[l], x[l])
            denom2 = 1 - np.inner(yk, yk)
            alpha = g(num/(denom1 * denom2))/denom1
            a += alpha
            b += alpha * x[l]
            c += alpha * np.inner(x[l], x[l])
        b2 = np.inner(b, b)
        yk = ((a + c - np.sqrt(clamp((a + c)**2 - 4 * b2)))/(2 * b2)) * b
    return yk

class PoincareKMeansFrechet(object):
    def __init__(self,n_dim=2,n_clusters=8,n_init=20,max_iter=300,c=1,tol=1e-8,verbose=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.labels_= None
        self.cluster_centers_ = None
        self.n_dim = n_dim
           

    def fit(self,X):
        n_samples = X.shape[0]
        self.inertia = None

        np.random.seed(11)
        for run_it in range(self.n_init):
            centroids = X[sample(range(n_samples), self.n_clusters),:]
            for it in range(self.max_iter):
                distances = self._get_distances_to_clusters(X,centroids)
                labels = np.argmin(distances,axis=1)

                new_centroids = np.zeros((self.n_clusters, self.n_dim))
                for i in range(self.n_clusters):
                    indices = np.where(labels==i)[0]
                    if len(indices)>0:
                        new_centroids[i,:] = self._hyperbolic_centroid(X[indices,:])
                    else:
                        new_centroids[i,:] = X[sample(range(n_samples), 1),:]
                m = np.ravel(centroids-new_centroids, order='K')
                diff = np.dot(m,m)
                centroids = new_centroids.copy()
                if(diff<self.tol):
                    break 
            distances = self._get_distances_to_clusters(X,centroids)
            labels = np.argmin(distances,axis=1)
            inertia = np.sum([np.sum(distances[np.where(labels==i)[0],i]**2) for i in range(self.n_clusters)])
            if (self.inertia == None) or (inertia<self.inertia):
                self.inertia = inertia
                self.labels_ = labels.copy()
                self.cluster_centers_ = centroids.copy()
                
            if self.verbose:
                print("Iteration: {} - Best Inertia: {}".format(run_it,self.inertia))
                          
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
    def predict(self,X):
        distances = self.transform(X)
        return np.argmin(distances,axis=1)
        
    def transform(self,X):
        return _get_distances_to_clusters(X, self.cluster_centers_)
    
    def _get_distances_to_clusters(self, X, clusters):
        #print(X.shape)
        n_samples, n_clusters = X.shape[0], clusters.shape[0]
        
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            centroid = np.tile(clusters[i,:],(n_samples,1))
            den1 = 1 - np.linalg.norm(X, axis=1)**2
            den2 = 1 - np.linalg.norm(centroid, axis=1)**2
            the_num = np.linalg.norm(X-centroid, axis=1)**2
            distances[:, i] = np.arccosh(1 + 2 * the_num/(den1 * den2))
        
        return distances
      
    def _hyperbolic_centroid(self,points):
        return frechet_mean(points,iterations=10)

def PoincareKMeansIter(params):
    it, n_dim, n_clusters, max_iter, tol, X = params
    n_samples = X.shape[0]
    np.random.seed(11)
    centroids = X[sample(range(n_samples), n_clusters),:]
    for it in tqdm(range(max_iter)):
        distances = get_distances_to_clusters(X,centroids)
        labels = np.argmin(distances,axis=1)

        new_centroids = np.zeros((n_clusters, n_dim))
        for i in range(n_clusters):
            indices = np.where(labels==i)[0]
            if len(indices)>0:
                new_centroids[i,:] = hyperbolic_centroid(X[indices,:])
            else:
                new_centroids[i,:] = X[sample(range(n_samples), 1),:]
        m = np.ravel(centroids-new_centroids, order='K')
        diff = np.dot(m,m)
        centroids = new_centroids.copy()
        if(diff<tol):
            break
                
    distances = get_distances_to_clusters(X,centroids)
    labels = np.argmin(distances,axis=1)
    inertia = np.sum([np.sum(distances[np.where(labels==i)[0],i]**2) for i in range(n_clusters)])
    return labels, centroids, inertia

def get_distances_to_clusters(X, clusters):
    #print(X.shape)
    n_samples, n_clusters = X.shape[0], clusters.shape[0]
        
    distances = np.zeros((n_samples, n_clusters))
    
    f = lambda a : None
    pool = mp.Pool(n_clusters)
    #for i in range(n_clusters):
    for i, _ in enumerate(pool.imap_unordered(f, range(n_clusters), 1)):
        centroid = np.tile(clusters[i,:],(n_samples,1))
        den1 = 1 - np.linalg.norm(X, axis=1)**2
        den2 = 1 - np.linalg.norm(centroid, axis=1)**2
        the_num = np.linalg.norm(X-centroid, axis=1)**2
        distances[:, i] = np.arccosh(1 + 2 * the_num/(den1 * den2))
        #sys.stderr.write('\rdone {0:%}'.format(i/len(range(n_clusters))))
    return distances

def hyperbolic_centroid(points):
    return frechet_mean(points,iterations=1)#10 tiempo

class PoincareKMeansParallel(object):
    def __init__(self,n_dim=2,n_clusters=8,n_init=20,max_iter=300,tol=1e-8,verbose=True, processes=1):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.labels_= None
        self.cluster_centers_ = None
        self.n_dim = n_dim
        self.processes = processes
           

    def fit(self,X):
        pool = mp.Pool(self.processes)
        z = zip(range(self.n_init), cycle([self.n_dim]), 
                           cycle([self.n_clusters]),
                           cycle([self.max_iter]),
                           cycle([self.tol]),
                           cycle([X]))
        inputs = [(it, n_dim, n_clusters, max_iter, tol, X) 
                  for (it, n_dim, n_clusters, max_iter, tol, X) in z]
        res = []
        for i, r in enumerate(pool.imap_unordered(PoincareKMeansIter, inputs, 1)):
            #sys.stderr.write('\rdone {0:%}'.format(i/len(inputs)))
            res.append(r)
        print("")
        pool.close()
        pool.join()
        
        idx = 0
        for i in tqdm(range(len(res))):
            if res[i][2] < res[idx][2]:
                idx = i
        self.labels_ = res[idx][0]
        self.cluster_centers_ = res[idx][1]
                          
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
    def predict(self,X):
        distances = self.transform(X)
        return np.argmin(distances,axis=1)
        
    def transform(self,X):
        return _get_distances_to_clusters(X, self.cluster_centers_)
    
    def _get_distances_to_clusters(self, X, clusters):
        #print(X.shape)
        n_samples, n_clusters = X.shape[0], clusters.shape[0]
        
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            centroid = np.tile(clusters[i,:],(n_samples,1))
            den1 = 1 - np.linalg.norm(X, axis=1)**2
            den2 = 1 - np.linalg.norm(centroid, axis=1)**2
            the_num = np.linalg.norm(X-centroid, axis=1)**2
            distances[:, i] = np.arccosh(1 + 2 * the_num/(den1 * den2))
        
        return distances
      
    def _hyperbolic_centroid(self,points):
        return frechet_mean(points,iterations=10)#10
    