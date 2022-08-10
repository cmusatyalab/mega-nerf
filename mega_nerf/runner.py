import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from mega_nerf.datasets.filesystem_dataset import FilesystemDataset
from mega_nerf.datasets.memory_dataset import MemoryDataset
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.models.model_utils import get_nerf, get_bg_nerf
from mega_nerf.ray_utils import get_rays, get_ray_directions
from mega_nerf.rendering import render_rays
from mega_nerf.sdf_utils import get_sdf_loss


class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)

        if hparams.ckpt_path is not None:
            # 从上一次训练中恢复随机数生成器状态, 并没有加载具体模型 (见 train())
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            random.setstate(checkpoint['random_state'])
        else:
            # 新一次训练，需要设置随机数生成器的种子
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams

        if 'RANK' in os.environ:
            """
            多卡训练, 区分主从机
            """
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None  # 实验根目录
            self.model_path = self.experiment_path / 'models' if self.is_master else None  # 模型 checkpoints 文件目录

        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        """
        加载数据集
        - coordinates.pt 中包含数据集的坐标原点和坐标系的 scale_factor
            - origin_drb: 坐标原点的坐标 (down-right-back), torch.Tensor, shape = (3)
            - scale_factor: 坐标系的 scale_factor, 使整个坐标系中所有相机位姿缩放到 [-1, 1], type=int
        - hparams.dataset_path: 数据集的路径，在 train.py 中定义
        """
        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor']
        main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        """
        - hparams.near (opts.py): 相机的近距离边界, 单位: m, 默认均为1
        - hparams.far (opts.py): 相机的远距离边界, 单位: m, 默认为 None, 这时会根据是否使用前后景划分自动设置
        - hparams.bg_nerf (opts.py): 是否使用背景 NeRF, 默认为 True
            - 当设置了 hparams.far 时，将该值除以 pose_scale_factor
            - 当未设置 hparams.far 且 hparams.bg_nerf 为 True 时,far = 1e5
            - 当未设置 hparams.far 且 hparams.bg_nerf 为 False 时,far = 2, 即整个场景会被归一化
        """
        self.near = hparams.near / self.pose_scale_factor

        if self.hparams.far is not None:
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf:
            self.far = 1e5
        else:
            self.far = 2

        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

        """
        将场景光线的高度范围根据 origin_drb 和 pose_scale_factor 进行归一化
        该参数是一个经验性的参数, 详见 https://github.com/cmusatyalab/mega-nerf/issues/6#issuecomment-1103187971
        """
        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in
                                   hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
        main_print('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
        main_print('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

        if self.ray_altitude_range is not None:
            assert self.ray_altitude_range[0] < self.ray_altitude_range[1]

        """
        导入 create_cluster_masks.py 生成的 mask
        - 检查 near, origin_drb, pose_scale_factor, ray_altitude_range 和这里的参数是否一致
        - params.pt 里面的内容：
            - origin_drb: 坐标原点的坐标 (down-right-back), torch.Tensor, shape = (3)
            - pose_scale_factor: 坐标系的 scale_factor, 使整个坐标系中所有相机位姿缩放到 [-1, 1], type=int
            - ray_altitude_range: 归一化到 [-1, 1] 的光线高度范围, type=list, [torch.FloatTensor, torch.FloatTensor]
            - near: 相机的近距离边界, 单位: m, 默认均为1, type=float
            - far: 相机的远距离边界, 单位: m, 默认为 2, 这个数据好像不会被使用 TODO: 检查
            - centroids: 各个类的中心点, type=torch.Tensor
                - shape=(n_clusters, 3), n_clusters = grid_x * grid_y
                - 每个中心点用一个 3D 坐标表示，坐标是归一化后的
            - grid_dim: 网格划分的维度, list[int, int], 分别为 x 和 y 轴上划分的块的数目, 即 [grid_x, grid_y]
                - 这个参数的设置也是经验性的, 详见 https://github.com/cmusatyalab/mega-nerf/issues/5
            - min_position: 所有相机位置范围的最小值, type=torch.Tensor, shape=(3)
            - max_position: 所有相机位置范围的最大值, type=torch.Tensor, shape=(3)
            - cluster_2d: type=bool TODO: 看看这个是干啥的
        """
        if self.hparams.cluster_mask_path is not None:
            cluster_params = torch.load(Path(self.hparams.cluster_mask_path).parent / 'params.pt', map_location='cpu')
            assert cluster_params['near'] == self.near
            assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
            assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

            if self.ray_altitude_range is not None:
                assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                       torch.FloatTensor(self.ray_altitude_range))), \
                    '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

        # 读取元信息
        self.train_items, self.val_items = self._get_image_metadata()
        main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        """
        重新计算了相机位置的范围, 不知道为什么不重用 params.pt 中的数据
        """
        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                max_position * self.pose_scale_factor + self.origin_drb))

        main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        """
        初始化 NeRF 模型
        """
        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)
        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                  output_device=int(os.environ['LOCAL_RANK']))

        if hparams.bg_nerf:
            """
            使用前后景分离的方式训练
            - 如果指定 hparams.ellipse_bounds, 则使用椭球将所有相机位置囊括起来，否则使用球形边界 (NeRF++)
            """
            self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
            if 'RANK' in os.environ:
                self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                         device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                         output_device=int(os.environ['LOCAL_RANK']))

            if hparams.ellipse_bounds:
                """
                使用椭球作为相机位姿的边界, 椭球的中心为相机位置的平均值, 将所有可能出现的相机位置都包含在椭球内
                """
                assert hparams.ray_altitude_range is not None

                if self.ray_altitude_range is not None:
                    ground_poses = camera_positions.clone()
                    ground_poses[:, 0] = self.ray_altitude_range[1]
                    air_poses = camera_positions.clone()
                    air_poses[:, 0] = self.ray_altitude_range[0]
                    used_positions = torch.cat([camera_positions, air_poses, ground_poses])
                else:
                    used_positions = camera_positions

                max_position[0] = self.ray_altitude_range[1]
                main_print('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                                 max_position))

                self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
                # 取相机位置的最大距离作为椭球的半径初值
                self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
                # 考虑光线的高度范围, 将椭球的半径扩大一些
                scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                    dim=-1).max()

                self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
            else:
                self.sphere_center = None
                self.sphere_radius = None

            main_print('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
        else:
            self.bg_nerf = None
            self.sphere_center = None
            self.sphere_radius = None
        # end __init__

    def train(self):
        self._setup_experiment_dir()

        # 自动混合精度计算, 通过 hparams.amp 指定
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        """
        初始化 optimizers, 使用 Adam 训练, 前后景模型使用两个不同的 optimizer
        """
        optimizers = {}
        optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            """
            从 checkpoint 恢复模型参数
            """
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']

            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)

            for key, optimizer in optimizers.items():
                optimizer_dict = optimizer.state_dict()
                optimizer_dict.update(checkpoint['optimizers'][key])
                optimizer.load_state_dict(optimizer_dict)
            discard_index = checkpoint['dataset_index'] if self.hparams.resume_ckpt_state else -1
        else:
            train_iterations = 0
            discard_index = -1

        """
        学习率衰减
        - ExponentialLR 规划器, 学习率衰减速度由 hparams.lr_decay_rate 和训练轮数指定
        """
        schedulers = {}
        for key, optimizer in optimizers.items():
            schedulers[key] = ExponentialLR(optimizer,
                                            gamma=self.hparams.lr_decay_factor ** (1 / self.hparams.train_iterations),
                                            last_epoch=train_iterations - 1)

        """
        加载数据集
        """
        if self.hparams.dataset_type == 'filesystem':
            """
            从磁盘中读取数据, 训练过程中需要将部分数据存储在一个缓冲区中 (chunk_dir)
            """
            # Let the local master write data to disk first
            # We could further parallelize the disk writing process by having all of the ranks write data,
            # but it would make determinism trickier
            if 'RANK' in os.environ and (not self.is_local_master):
                dist.barrier()

            dataset = FilesystemDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                        self.hparams.center_pixels, self.device,
                                        [Path(x) for x in sorted(self.hparams.chunk_paths)], self.hparams.num_chunks,
                                        self.hparams.train_scale_factor, self.hparams.disk_flush_size)
            if self.hparams.ckpt_path is not None and self.hparams.resume_ckpt_state:
                dataset.set_state(checkpoint['dataset_state'])
            if 'RANK' in os.environ and self.is_local_master:
                dist.barrier()
        elif self.hparams.dataset_type == 'memory':
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device)
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None

        while train_iterations < self.hparams.train_iterations:
            # If discard_index >= 0, we already set to the right chunk through set_state
            if self.hparams.dataset_type == 'filesystem' and discard_index == -1:
                dataset.load_chunk()

            """
            创建 DataLoader
            """
            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
            else:
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                         pin_memory=True)

            for dataset_index, item in enumerate(data_loader):
                if dataset_index <= discard_index:
                    continue

                discard_index = -1

                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    if self.hparams.appearance_dim > 0:
                        image_indices = item['img_indices'].to(self.device, non_blocking=True)
                    else:
                        image_indices = None

                    metrics, bg_nerf_rays_present = self._training_step(
                        item['rgbs'].to(self.device, non_blocking=True),
                        item['depths'].to(self.device, non_blocking=True),
                        item['rays'].to(self.device, non_blocking=True),
                        image_indices)

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                raise Exception('Train metrics not finite: {}'.format(metrics))

                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)

                scaler.scale(metrics['loss']).backward()

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    else:
                        scaler.step(optimizer)

                scaler.update()

                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    for key, value in metrics.items():
                        self.writer.add_scalar('train/{}'.format(key), value, train_iterations)

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                              dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

                if train_iterations > 0 and train_iterations % self.hparams.val_interval == 0:
                    self._run_validation(train_iterations)

                if train_iterations >= self.hparams.train_iterations:
                    break

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

        if self.hparams.cluster_mask_path is None:
            val_metrics = self._run_validation(train_iterations)
            self._write_final_metrics(val_metrics)

    def eval(self):
        self._setup_experiment_dir()
        val_metrics = self._run_validation(0)
        self._write_final_metrics(val_metrics)

    def _write_final_metrics(self, val_metrics: Dict[str, float]) -> None:
        if self.is_master:
            with (self.experiment_path / 'metrics.txt').open('w') as f:
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))

            self.writer.flush()
            self.writer.close()

    def _setup_experiment_dir(self) -> None:
        """
        初始化实验运行目录
        - 保存 hparams
        - 保存运行指令
        - 保存 image_indices
        - 初始化 Tensorboard Writer
        """
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None

        if 'RANK' in os.environ:
            dist.barrier()
        # end _setup_experiment_dir

    def _training_step(self, rgbs: torch.Tensor, depths: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor]) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                    bg_nerf=self.bg_nerf,
                                                    rays=rays,
                                                    image_indices=image_indices,
                                                    hparams=self.hparams,
                                                    sphere_center=self.sphere_center,
                                                    sphere_radius=self.sphere_radius,
                                                    get_depth=True,
                                                    get_depth_variance=True,
                                                    get_bg_fg_rgb=False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            depth_variance = results[f'depth_variance_{typ}'].mean()

        metrics = {
            'psnr': psnr_,
            'depth_variance': depth_variance,
        }

        photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean') * self.hparams.photo_weight
        depth_loss = self.hparams.depth_weight * F.mse_loss(results[f'depth_{typ}'].view(-1, 1), depths.view(-1, 1), reduction='mean')
        fs_loss, tr_loss = get_sdf_loss(results[f'zvals_{typ}'], results[f'raw_sigma_{typ}'], depths)
        sdf_loss = (fs_loss + tr_loss) * self.hparams.sdf_weight
        metrics['photo_loss'] = photo_loss
        metrics['depth_mse_loss'] = depth_loss
        metrics['sdf_loss'] = sdf_loss
        metrics['loss'] = photo_loss + depth_loss + sdf_loss

        if self.hparams.use_cascade and typ != 'coarse':
            coarse_loss = F.mse_loss(results['rgb_coarse'], rgbs, reduction='mean')

            metrics['coarse_loss'] = coarse_loss
            metrics['loss'] += coarse_loss
            metrics['loss'] /= 2

        return metrics, bg_nerf_rays_present

    def _run_validation(self, train_index: int) -> Dict[str, float]:
        with torch.inference_mode():
            self.nerf.eval()

            val_metrics = defaultdict(float)
            base_tmp_path = None
            try:
                if 'RANK' in os.environ:
                    base_tmp_path = Path(self.hparams.exp_name) / os.environ['TORCHELASTIC_RUN_ID']
                    metric_path = base_tmp_path / 'tmp_val_metrics'
                    image_path = base_tmp_path / 'tmp_val_images'

                    world_size = int(os.environ['WORLD_SIZE'])
                    indices_to_eval = np.arange(int(os.environ['RANK']), len(self.val_items), world_size)
                    if self.is_master:
                        base_tmp_path.mkdir()
                        metric_path.mkdir()
                        image_path.mkdir()
                    dist.barrier()
                else:
                    indices_to_eval = np.arange(len(self.val_items))

                for i in main_tqdm(indices_to_eval):
                    metadata_item = self.val_items[i]
                    viz_rgbs = metadata_item.load_image().float() / 255.
                    viz_gt_depths = metadata_item.load_depth_image().float()

                    results, _ = self.render_image(metadata_item)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
                    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()

                    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))

                    metric_key = 'val/psnr/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_psnr, train_index)
                    else:
                        torch.save({'value': val_psnr, 'metric_key': metric_key, 'agg_key': 'val/psnr'},
                                   metric_path / 'psnr-{}.pt'.format(i))

                    val_metrics['val/psnr'] += val_psnr

                    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

                    metric_key = 'val/ssim/{}'.format(i)
                    if self.writer is not None:
                        self.writer.add_scalar(metric_key, val_ssim, train_index)
                    else:
                        torch.save({'value': val_ssim, 'metric_key': metric_key, 'agg_key': 'val/ssim'},
                                   metric_path / 'ssim-{}.pt'.format(i))

                    val_metrics['val/ssim'] += val_ssim

                    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)

                    for network in val_lpips_metrics:
                        agg_key = 'val/lpips/{}'.format(network)
                        metric_key = '{}/{}'.format(agg_key, i)
                        if self.writer is not None:
                            self.writer.add_scalar(metric_key, val_lpips_metrics[network], train_index)
                        else:
                            torch.save(
                                {'value': val_lpips_metrics[network], 'metric_key': metric_key, 'agg_key': agg_key},
                                metric_path / 'lpips-{}-{}.pt'.format(network, i))

                        val_metrics[agg_key] += val_lpips_metrics[network]

                    viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                    viz_depth = results[f'depth_{typ}']
                    if f'fg_depth_{typ}' in results:
                        to_use = results[f'fg_depth_{typ}'].view(-1)
                        while to_use.shape[0] > 2 ** 24:
                            to_use = to_use[::2]
                        ma = torch.quantile(to_use, 0.95)

                        viz_depth = viz_depth.clamp_max(ma)

                    img = Runner._create_result_image(viz_rgbs, viz_result_rgbs, viz_gt_depths, viz_depth)

                    if self.writer is not None:
                        self.writer.add_image('val/{}'.format(i), T.ToTensor()(img), train_index)
                    else:
                        img.save(str(image_path / '{}.jpg'.format(i)))

                    if self.hparams.bg_nerf:
                        if f'bg_rgb_{typ}' in results:
                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              viz_gt_depths,
                                                              results[f'bg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_bg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_bg.jpg'.format(i)))

                            img = Runner._create_result_image(viz_rgbs,
                                                              results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],
                                                                                            viz_rgbs.shape[1],
                                                                                            3).cpu(),
                                                              viz_gt_depths,
                                                              results[f'fg_depth_{typ}'])

                            if self.writer is not None:
                                self.writer.add_image('val/{}_fg'.format(i), T.ToTensor()(img), train_index)
                            else:
                                img.save(str(image_path / '{}_fg.jpg'.format(i)))

                    del results

                if 'RANK' in os.environ:
                    dist.barrier()
                    if self.writer is not None:
                        for metric_file in metric_path.iterdir():
                            metric = torch.load(metric_file, map_location='cpu')
                            self.writer.add_scalar(metric['metric_key'], metric['value'], train_index)
                            val_metrics[metric['agg_key']] += metric['value']
                        for image_file in image_path.iterdir():
                            img = Image.open(str(image_file))
                            self.writer.add_image('val/{}'.format(image_file.stem), T.ToTensor()(img), train_index)

                        for key in val_metrics:
                            avg_val = val_metrics[key] / len(self.val_items)
                            self.writer.add_scalar('{}/avg'.format(key), avg_val, 0)

                    dist.barrier()

                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)

            return val_metrics

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        directions = get_ray_directions(metadata.W,
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True)  # (H*W, 8)
            image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                if self.hparams.appearance_dim > 0 else None
            results = {}

            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True)

                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []

                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)

            return results, rays

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, gt_depth: torch.Tensor
                            , result_depths: torch.Tensor) -> Image:
        depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
        gt_depth_vis = Runner.visualize_scalars(torch.log(gt_depth + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
        images = (rgbs * 255, result_rgbs * 255)
        depth = (gt_depth_vis, depth_vis)
        ret = np.concatenate([np.concatenate(images, axis=1), np.concatenate(depth, axis=1)], axis=0).astype(np.uint8)
        return Image.fromarray(ret)

    @staticmethod
    def visualize_scalars(scalar_tensor: torch.Tensor) -> np.ndarray:
        to_use = scalar_tensor.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi = torch.quantile(to_use, 0.05)
        ma = torch.quantile(to_use, 0.95)

        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        """
        从 hparams.dataset_path 指定的数据集位置读入元信息
        """
        dataset_path = Path(self.hparams.dataset_path)

        train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
        train_paths = [train_path_candidates[i] for i in
                       range(0, len(train_path_candidates), self.hparams.train_every)]

        val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))
        train_paths += val_paths
        train_paths.sort(key=lambda x: x.name)
        val_paths_set = set(val_paths)

        image_indices = {}
        for i, train_path in enumerate(train_paths):
            image_indices[train_path.name] = i

        """
        这里训练集中也会有验证集的图片, 用于训练 per-image 的 appearance embedding
        - 详见: https://github.com/cmusatyalab/mega-nerf/issues/18
        """
        train_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set) for x
            in train_paths]
        val_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, True) for x in val_paths]

        return train_items, val_items

    def _get_metadata_item(self, metadata_path: Path, image_index: int, scale_factor: int,
                           is_val: bool) -> ImageMetadata:
        """
        从元数据文件中读入元信息
        - 确认对应的图片存在
        - 加载对应的元信息 pt 文件
        - 将内参按照 scale_factor 缩放, 确保图片大小能够被 scale_factor 整除

        Input:
            metadata_path: 元数据文件路径, 地址是训练集或验证集中 rgbs/metadata/{image_name}.pt
            image_index: 图像索引
            scale_factor: 图像缩放比例
            is_val: 是否为验证集
        Return:
            ImageMetadata: 元信息
        """
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        assert image_path.exists()

        depth_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'depthvis' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                depth_path = candidate
                break

        assert depth_path.exists()
        """
        加载对应的元信息 pt 文件
        - H, W: 图片的高和宽
        - c2w: 图片的变换矩阵, torch.Tensor, shape=(3, 4)
            - 旋转矩阵是 (right, up, backwards)
            - 平移是 (down, right, backwards)
            - 详见： https://github.com/cmusatyalab/mega-nerf/issues/3
        - intrinsics: 图片的内参, torch.Tensor, (fx, fy, cx, cy)
        - distortion: torch.Tensor, shape=(4)
        """
        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        assert metadata['W'] % scale_factor == 0
        assert metadata['H'] % scale_factor == 0

        """
        加载对应的 masks 文件夹中的 metadata
        - 该文件是一个 ZipFile, 需要用 ZipFile.open 打开后用 torch.load 读取
        - 详见: scripts/create_cluster_masks.py
        - 其中的 mask 是一个 torch.Tensor, shape=(H, W), dtype=torch.bool

        masks 的位置可以通过 hparams.cluster_mask_path 指定, 也可以放在数据集根目录下的 masks 目录中
        如果都没有指定, 则 masks 为空, 因此 masks 不是必须的
        """
        dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_print('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
            mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
        elif dataset_mask.exists():
            if image_index == 0:
                main_print('Using dataset mask path: {}'.format(dataset_mask.parent))
            mask_path = dataset_mask
        else:
            mask_path = None

        """
        当在 hparams 中指定 all_val 为真时, 不使用 masks 进行 inference
        """
        return ImageMetadata(image_path, depth_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                             intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val)

    def _get_experiment_path(self) -> Path:
        """
        创建新的实验目录，并返回实验目录的路径
        在输入参数中 exp_name 中新建一个 version 目录，名称为版本号
        """
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path
