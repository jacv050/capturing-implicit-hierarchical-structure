import sys
sys.path.append(".")
sys.path.append("..")
import os
import datetime
import json
import argparse
from tempfile import mkdtemp
from collections import defaultdict
import subprocess
import math
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import time
import pickle

from utils import Logger, Timer, save_model, save_vars, probe_infnan
import objectives
import models

runId = datetime.datetime.now().isoformat().replace(':','_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

### General
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--model', type=str, metavar='M', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', choices=['Euclidean', 'PoincareBall'])
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=0, help='print objective values every value (if positive)')
parser.add_argument('--skip-test', action='store_true', default=False, help='skip test dataset computations')
parser.add_argument('--eval', action='store_true', default=False, help='run inference')

### Dataset
parser.add_argument('--data-params', nargs='+', default=[], help='parameters which are passed to the dataset loader')
parser.add_argument('--data-size', type=int, nargs='+', default=[], help='size/shape of data observations')
parser.add_argument('--np_zeropadding', default=False, action='store_true', help='Fill p and n with zero padding.')
parser.add_argument('--no_padding', default=False, action='store_true', help='No padding test set for different sizes.')

### Metric & Plots
parser.add_argument('--iwae-samples', type=int, default=0, help='number of samples to compute marginal log likelihood estimate')

### Optimisation
parser.add_argument('--obj', type=str, default='vae', help='objective to minimise (default: vae)')
parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs to train (default: 50)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
parser.add_argument('--lr', type=float, default=1e-4, help='learnign rate for optimser (default: 1e-4)')
parser.add_argument('--clip', type=float, default=-1., help='max norm size for gradient clipping')

## Objective
parser.add_argument('--K', type=int, default=1, metavar='K',  help='number of samples to estimate ELBO (default: 1)')
parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='coefficient of beta-VAE (default: 1.0)')
parser.add_argument('--analytical-kl', action='store_true', default=False, help='analytical kl when possible')
parser.add_argument('--triplet-loss', action='store_true', default=False, help='add triplet loss')
parser.add_argument('--triplet-weight',  type=float, default=1e3)
parser.add_argument('--triplet-loss-dist', action='store_true', default=False, help='triplet loss with dist')

### Model
parser.add_argument('--latent-dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--c', type=float, default=1., help='curvature')
parser.add_argument('--posterior', type=str, default='WrappedNormal', help='posterior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])

## Architecture
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100, help='number of hidden layers dimensions (default: 100)')
parser.add_argument('--nl', type=str, default='ReLU', help='non linearity')
parser.add_argument('--enc', type=str, default='Wrapped', help='allow to choose different implemented encoder',
                    choices=['Linear', 'Wrapped', 'Mob', 'WrappedConv', 'LinearConv', 'WrappedVideoConv'])
parser.add_argument('--dec', type=str, default='Wrapped', help='allow to choose different implemented decoder',
                    choices=['Linear', 'Wrapped', 'Geo', 'Mob', 'WrappedConv', 'LinearConv', 'GyroConv', 'GyroVideoConv', 'LinearVideoConv'])

## Prior
parser.add_argument('--prior-iso', action='store_true', default=False, help='isotropic prior')
parser.add_argument('--prior', type=str, default='WrappedNormal', help='prior distribution',
                    choices=['WrappedNormal', 'RiemannianNormal', 'Normal'])
parser.add_argument('--prior-std', type=float, default=1., help='scale stddev by this value (default:1.)')
parser.add_argument('--learn-prior-std', action='store_true', default=False)

### Technical
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.prior_iso = args.prior_iso or args.posterior == 'RiemannianNormal'

# Choosing and saving a random seed for reproducibility
if args.seed == 0: args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
print('seed', args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# Create directory for experiment if necessary
directory_name = os.path.join(args.save_dir, 'hyperbolic-model-checkpoints/{}'.format(args.name))
if args.name != '.':
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    runPath = directory_name
else:
    runPath = directory_name
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:', runId)

# Save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    fp.write(git_hash.decode('utf-8') + command)
torch.save(args, '{}/args.rar'.format(runPath))

# Initialise model, optimizer, dataset loader and loss function
modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))

train_loader, test_loader = model.getDataLoaders(args.batch_size, True, device, *args.data_params)
loss_function = getattr(objectives, args.obj + '_objective')


def train(epoch, agg):
    model.train()
    
    b_loss, b_recon, b_kl, b_triplet = 0., 0., 0., 0.
    for i, (parent, positive_child, negative_child, target_norm) in enumerate(tqdm(train_loader)):
        parent = parent.to(device)
        positive_child = positive_child.to(device)
        negative_child = negative_child.to(device)
        
        optimizer.zero_grad()
        
        is_poincare = False
        if args.manifold == 'PoincareBall':
            is_poincare = True
        
        if args.triplet_loss:
            qz_x, px_z, lik, kl, triplet, loss = loss_function(model, parent, positive_child, negative_child, K=args.K, beta=args.beta, components=True, analytical_kl=args.analytical_kl, triplet_loss=args.triplet_loss, triplet_loss_dist=args.triplet_loss_dist, is_poincare=is_poincare, triplet_weight=args.triplet_weight)
        else:
            qz_x, px_z, lik, kl, loss = loss_function(model, data, K=args.K, beta=args.beta, components=True, analytical_kl=args.analytical_kl)
        
        probe_infnan(loss, "Training loss:")
        #t1 = time.time()
        loss.backward()
        #t2 = time.time()
        #print("{}".format(t2-t1))
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()

        b_loss += loss.item()
        b_recon += -lik.mean(0).sum().item()
        b_kl += kl.sum(-1).mean(0).sum().item()
        #print(kl.sum(-1).mean(0).sum().item()/512)
        
        if args.triplet_loss:
            b_triplet += triplet.item()

    agg['train_loss'].append(b_loss / len(train_loader.dataset))
    agg['train_recon'].append(b_recon / len(train_loader.dataset))
    agg['train_kl'].append(b_kl / len(train_loader.dataset))
    
    if args.triplet_loss:
        agg['train_triplet'].append(b_triplet / len(train_loader.dataset))
        
    if epoch % 1 == 0 and args.triplet_loss:
        print('====> Epoch: {:03d} Loss: {:.2f} Recon: {:.2f} KL: {:.2f} Triplet: {:.4f}'.format(epoch, agg['train_loss'][-1], agg['train_recon'][-1], agg['train_kl'][-1], agg['train_triplet'][-1]))
    elif epoch % 1 == 0:
        print('====> Epoch: {:03d} Loss: {:.2f} Recon: {:.2f} KL: {:.2f}'.format(epoch, agg['train_loss'][-1], agg['train_recon'][-1], agg['train_kl'][-1]))


def test(epoch, agg):
    model.eval()
    b_loss, b_mlik = 0., 0.
    b_kl, b_recon = 0.0, 0.0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            #print(loss_function(model, data, K=args.K, beta=args.beta, components=True))
            qz_x, px_z, lik, kl, loss = loss_function(model, data, K=args.K, beta=args.beta, components=True)
            
            if epoch == args.epochs and args.iwae_samples > 0:
                mlik = objectives.iwae_objective(model, data, K=args.iwae_samples)
                b_mlik += mlik.sum(-1).item()
            b_loss += loss.item()
            
            b_kl += kl.sum(-1).mean(0).sum().item()
            b_recon += -lik.mean(0).sum().item()
            #if i == 0: model.reconstruct(data, runPath, epoch)

    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    agg['test_mlik'].append(b_mlik / len(test_loader.dataset))
    agg['test_kl'].append(b_kl / len(test_loader.dataset))
    agg['test_recon'].append(b_recon / len(test_loader.dataset))
    print('====>             Test loss: {:.4f} mlik: {:.4f} recon: {:.4f} KL: {:.2f}'.format(agg['test_loss'][-1], agg['test_mlik'][-1], agg['test_recon'][-1], agg['test_kl'][-1]))
    
    
def evaluate():
    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print("Checkpoint loaded...")
    model.eval()
    all_mus = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(test_loader)):
            data = data.to(device)            
            mus = model.getMus(data, runPath)
            mus = mus.data.cpu().numpy()
            all_mus.append(mus)
    all_mus = np.concatenate(all_mus) 

    save_path = os.path.join(args.save_dir, 'hyperbolic-mus-inference/{}'.format(args.name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pickle.dump(all_mus, open(save_path + '/mus.p', 'wb'))


if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        if args.eval:
            print('Starting testing...')
            evaluate()

        else:
            print('Starting training...')
            model.init_last_layer_bias(train_loader)
            for epoch in range(1, args.epochs + 1):
                train(epoch, agg)
                if args.save_freq == 0 or epoch % args.save_freq == 0:
                    if not args.skip_test: test(epoch, agg) 

                save_model(model, runPath + '/model.rar')
                #save_vars(agg, runPath + '/losses.rar')

            print('p(z) params:')
            print(model.pz_params) 
