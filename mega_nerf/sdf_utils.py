import io
from math import trunc
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from PIL import Image

def get_gt_sdf_masks(z_vals, gt_depth, truncation):
    """
    Inputs:
        z_vals: (batch_size, n_samples)
        gt_depth: (batch_size, 1)
        truncation: float
    Returns:
        front_mask: (batch_size, n_samples), 1 if point is in front (z < depth - tr)
        back_mask: (batch_size, n_samples), 1 if point is in back (z > depth + tr)
        sdf_mask: (batch_size, n_samples), 1 if point is in truncation range
                  i.e. (z - depth) < tr < (z - depth) > -tr
    """
    # before truncation 
    front_mask = torch.where(z_vals < (gt_depth - truncation), 
                            torch.ones_like(z_vals),
                            torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (gt_depth + truncation),
                            torch.ones_like(z_vals),
                            torch.zeros_like(z_vals))
    # sdf region
    sdf_mask = (1.0 - front_mask)
    sdf_mask *= (1.0 - back_mask)
    return front_mask, back_mask, sdf_mask

def get_gt_sdf(z_vals, gt_depth, truncation, front_mask, back_mask, sdf_mask):
    """
    Inputs:
        z_vals: (batch_size, n_samples)
        gt_depth: (batch_size, 1)
        truncation: float
    """
    fs_sdf = (front_mask - back_mask) * torch.ones_like(z_vals)
    tr_sdf = sdf_mask * (gt_depth - z_vals) / truncation
    return fs_sdf + tr_sdf

def plot_sdf_gt_with_predicted(z_vals, gt_sdf, predicted_sdf, gt_depth, truncation):
    """
    Inputs:
        z_vals: (1, n_samples)
        gt_sdf: (1, n_samples)
        predicted_sdf: (1, n_samples)
    """
    plt.figure(figsize=(6, 6))
    z_vals = z_vals.detach().numpy().reshape(-1)
    gt_sdf = gt_sdf.detach().numpy().reshape(-1)
    predicted_sdf = predicted_sdf.detach().numpy().reshape(-1)
    plt.plot(z_vals, gt_sdf, label='gt_sdf')
    plt.plot(z_vals, predicted_sdf, label='predicted_sdf')
    plt.plot(z_vals, np.zeros_like(z_vals), '--')
    plt.plot(gt_depth * np.ones_like(z_vals), np.linspace(-truncation, truncation, z_vals.shape[0]), '--', label='gt_depth')
    plt.legend()
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype='uint8').reshape(h, w, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes('RGBA', (w, h), buf.tostring())
    plt.close()
    return np.asarray(image)[..., :3].transpose(2, 0, 1)

def sdf2weight(z_vals, sdf, truncation=0.05):
    """
    from Neural-RGB-D
    Inputs:
        sdf: (N_rays, N_samples)
    Outputs:
        weights: (N_rays, N_samples)
    """
    # compute raw weights according to the paper
    weights = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)
    # if there exists multiple surface, we should only keep the first one
    # compute the zero-crossing
    signs = sdf[:, 1:] * sdf[:, :-1]
    mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
    inds = torch.argmax(mask, axis=1)
    inds = inds[..., None]
    z_min = torch.gather(z_vals, 1, inds) # The first surface
    mask = torch.where(z_vals < z_min + truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

    weights = weights * mask
    return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)

def get_sdf_loss(z_vals, predicted_sdf, gt_depth, truncation=0.05):
    """
    calculate SDF losses, consists of two parts:
    1. freespace sdf loss, includes before/after truncation region
    2. truncation loss
    in this function, we first compute masks for the truncation region
    and compute losses respectively

    Inputs:
        z_vals: (batch_size, n_samples)
        predicted_sdf: (batch_size, n_samples)
        gt_depth: (batch_size, 1)
    """
    mse = lambda x, y: torch.mean((x - y) ** 2)
    gt_depth = gt_depth[:, None]
    front_mask, back_mask, sdf_mask = get_gt_sdf_masks(z_vals, gt_depth, truncation)
    front_samples = torch.count_nonzero(front_mask)
    sdf_samples = torch.count_nonzero(sdf_mask)
    
    gt_sdf = get_gt_sdf(z_vals, gt_depth, truncation, front_mask, back_mask, sdf_mask)

    return mse(predicted_sdf * front_mask, gt_sdf * front_mask) / front_samples, \
        mse(predicted_sdf * sdf_mask, gt_sdf * sdf_mask) / sdf_samples