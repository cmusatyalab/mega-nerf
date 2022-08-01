from argparse import Namespace

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from mega_nerf.models.cascade import Cascade
from mega_nerf.models.mega_nerf import MegaNeRF
from mega_nerf.models.nerf import NeRF, ShiftedSoftplus


def get_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    """
    生成 NeRF-W 模型 (前景)
    Inputs:
    - hparams: 训练参数
    - appearance_count: per-image appearance embedding 的个数, 即图片个数
    Returns:
    - nerf: NeRF-W 模型
    """
    return _get_nerf_inner(hparams, appearance_count, hparams.layer_dim, 3, 'model_state_dict')


def get_bg_nerf(hparams: Namespace, appearance_count: int) -> nn.Module:
    """
    生成 NeRF-W 模型 (背景)
    Inputs:
    - hparams: 训练参数
    - appearance_count: per-image appearance embedding 的个数, 即图片个数
    Returns:
    - nerf: NeRF-W 模型 (背景, 输入坐标为 4 维, xyz + 归一化逆深度)
    """
    return _get_nerf_inner(hparams, appearance_count, hparams.bg_layer_dim, 4, 'bg_model_state_dict')


def _get_nerf_inner(hparams: Namespace, appearance_count: int, layer_dim: int, xyz_dim: int,
                    weight_key: str) -> nn.Module:
    """
    生成 NeRF-W 模型
    - 如果 hparams.container_path 不为空, 即加载合并过的模型
    - 如果 hparams.use_cascade 是 True, 即使用粗采样-精采样级联模型
    Inputs:
    - hparams: 训练参数
    - appearance_count: per-image appearance embedding 的个数, 即图片个数
    - layer_dim: 每层的维度
    - xyz_dim: 输入坐标的维度, 3 维或 4 维, 如果为 4 维, 则输入坐标为 xyz + 归一化逆深度
    - weight_key: 模型权重的 key, 用于加载/保存模型权重
    Return:
    - nerf: NeRF-W 模型
    """
    if hparams.container_path is not None:
        container = torch.jit.load(hparams.container_path, map_location='cpu')
        if xyz_dim == 3:
            return MegaNeRF([getattr(container, 'sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, False, container.cluster_2d)
        else:
            return MegaNeRF([getattr(container, 'bg_sub_module_{}'.format(i)) for i in range(len(container.centroids))],
                            container.centroids, hparams.boundary_margin, True, container.cluster_2d)
    elif hparams.use_cascade:
        nerf = Cascade(
            _get_single_nerf_inner(hparams, appearance_count,
                                   layer_dim if xyz_dim == 4 else layer_dim,    # 废话代码学 ?
                                   xyz_dim),
            _get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim))
    elif hparams.train_mega_nerf is not None:
        centroid_metadata = torch.load(hparams.train_mega_nerf, map_location='cpu')
        centroids = centroid_metadata['centroids']
        nerf = MegaNeRF(
            [_get_single_nerf_inner(hparams, appearance_count, layer_dim, xyz_dim) for _ in
             range(len(centroids))], centroids, 1, xyz_dim == 4, centroid_metadata['cluster_2d'], True)
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
                hparams.affine_appearance,
                appearance_count,
                rgb_dim,
                xyz_dim,
                ShiftedSoftplus() if hparams.shifted_softplus else nn.ReLU())
