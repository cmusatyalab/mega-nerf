from argparse import Namespace

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from mega_nerf.models.cascade import Cascade
from mega_nerf.models.mega_nerf import MegaNeRF
from mega_nerf.models.nerf import NeRF, ShiftedSoftplus


def get_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.layer_dim, 3, 'model_state_dict')


def get_bg_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    return _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict')


def _get_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int,
                    weight_key: str) -> nn.Module:
    if hparams.container_path is not None:
        container = torch.jit.load(hparams.container_path, map_location='cpu')
        if xyz_dim == 3:
            return MegaNeRF([getattr(container, 'sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, False)
        else:
            return MegaNeRF([getattr(container, 'bg_sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, True)
    elif hparams.use_cascade:
        nerf = Cascade(_get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim),
                       _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim))
    elif hparams.train_mega_nerf is not None:
        centroids = torch.load(hparams.train_mega_nerf, map_location='cpu')['centroids']
        nerf = MegaNeRF(
            [_get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim) for _ in
             range(len(centroids))], centroids, hparams.boundary_margin, xyz_dim == 4, True)
    else:
        nerf = _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim)

    if hparams.ckpt_path is not None:
        state_dict = torch.load(hparams.ckpt_path, map_location='cpu')[weight_key]
        consume_prefix_in_state_dict_if_present(state_dict, prefix='module.')

        model_dict = nerf.state_dict()
        model_dict.update(state_dict)
        nerf.load_state_dict(model_dict)

    return nerf


def _get_single_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int) -> nn.Module:
    rgb_dim = 3 * ((hparams.sh_deg + 1) ** 2) if hparams.sh_deg is not None else 3

    return NeRF(hparams.pos_xyz_dim,
                hparams.pos_dir_dim,
                hparams.layers,
                hparams.skip_layers,
                layer_dim,
                hparams.appearance_dim,
                appearance_count,
                rgb_dim,
                xyz_dim,
                ShiftedSoftplus() if hparams.shifted_softplus else nn.ReLU())
