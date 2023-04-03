import torch
import torch.distributions as dist
from numpy import prod
from pvae.utils import has_analytic_kl, log_mean_exp
import torch.nn.functional as F
from geoopt.manifolds.poincare.math import mobius_add, dist


def vae_objective(model, x, positive_child=None, negative_child=None, K=1, beta=1.0, components=False, analytical_kl=False, target_norm=None, norm_loss=False, triplet_loss=False, triplet_loss_dist=False, triplet_norm_loss=False, is_poincare=True, margin=0.2, triplet_margin=0.4, triplet_weight=1e3, **kwargs):

    """Computes E_{p(x)}[ELBO] """
    if triplet_loss:
        qz_x, px_z, zs, parent_mu, positive_child_mu, negative_child_mu = model(x, positive_child, negative_child, K)
    else:
        qz_x, px_z, zs = model(x, positive_child, negative_child, K)
        
        
    _, B, D = zs.size()
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)

    pz = model.pz(*model.pz_params)
    kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1) if \
        has_analytic_kl(type(qz_x), model.pz) and analytical_kl else \
        qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)
    
    obj = -lpx_z.mean(0).sum() + beta * kld.mean(0).sum()
    
    if triplet_loss:
        if is_poincare:
            if triplet_loss_dist:
                distance_positive = dist(parent_mu, positive_child_mu, keepdim=True).pow(2).sum(1)
                distance_negative = dist(parent_mu, negative_child_mu, keepdim=True).pow(2).sum(1)
            else:
                distance_positive = mobius_add(parent_mu, -positive_child_mu).pow(2).sum(1)
                distance_negative = mobius_add(parent_mu, -negative_child_mu).pow(2).sum(1)
        else:
            distance_positive = (parent_mu - positive_child_mu).pow(2).sum(1)
            distance_negative = (parent_mu - negative_child_mu).pow(2).sum(1)
        triplet_losses = F.relu(distance_positive - distance_negative + margin)
        triplet = triplet_losses.mean()
    
        triplet = triplet * triplet_weight
        obj += triplet
        return (qz_x, px_z, lpx_z, kld, triplet, obj) if components else obj
    else:
        return (qz_x, px_z, lpx_z, kld, obj) if components else obj

def _iwae_objective_vec(model, x, K):
    """Helper for IWAE estimate for log p_\theta(x) -- full vectorisation."""
    qz_x, px_z, zs = model(x, K)
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x.expand(zs.size(0), *x.size())).view(flat_rest).sum(-1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    obj = lpz.squeeze(-1) + lpx_z.view(lpz.squeeze(-1).shape) - lqz_x.squeeze(-1)
    return -log_mean_exp(obj).sum()


def iwae_objective(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    Appropriate negation (for minimisation) happens in the helper
    """
    split_size = int(x.size(0) / (K * prod(x.size()) / (3e7)))  # rough heuristic
    if split_size >= x.size(0):
        obj = _iwae_objective_vec(model, x, K)
    else:
        obj = 0
        for bx in x.split(split_size):
            obj = obj + _iwae_objective_vec(model, bx, K)
    return obj
