from logging import warning
from math import floor
import os
from pathlib import Path
import argparse
from typing import List
import numpy as np
import torch
import cv2
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import transformations
import pfm_utils

def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='path to source dataset directory')
    parser.add_argument('--depth_tracks', type=str, nargs='+', required=True, help='list of depth track names, e.g you have two depth tracks in depth_aligned/ and depth_offset/, then put "aligned offset" in this argument')
    parser.add_argument('--intrinsics', type=float, nargs=4, required=True, help='instrinsic tuple (fx, fy, sx, sy)')
    parser.add_argument('--pose_scale_factor', type=int, help='pose scaling factors, make sure your pose positions are in [-1, -1, -1] to [1, 1, 1]')
    parser.add_argument('--output_path', type=str, default='./output', help='path to target dataset directory')
    parser.add_argument('--num_val', type=int, default=10, help='number of validation images')
    parser.add_argument('--sample_rate', type=float, default=1, help='percentage of data you wanna keep')
    parser.add_argument('--pose_storage_format', choices=['trajectory', 'frame'], default='frame', help='store all pose vectors together as a trajectory-wised format, or store each pose vector in a seperate file as a frame-wised format')
    return parser.parse_args()
    

hparams = _get_opts()
basepath = Path(hparams.data_path)
mega_path = Path(hparams.output_path)

assert basepath.exists(), "dataset not exist in " + str(basepath)
assert (basepath / 'rgb').exists(), "rgb image dir not exists"
assert len(hparams.intrinsics) == 4, f"intrinsics {hparams.intrinsics} illegal."

K = np.eye(4)
(K[0, 0], K[1,1], K[0, 2], K[1, 2]) = hparams.intrinsics
K_inv = np.linalg.inv(K)

if not os.path.exists(mega_path):
    os.makedirs(mega_path)

distortion = np.array([0,0,0,0])
num_val = hparams.num_val
depth_tracks = hparams.depth_tracks

# READ DIRECTORY

rgb_names = [x for x in Path(basepath / 'rgb').iterdir() if x.suffix == '.jpg']
rgb_names = sorted(rgb_names, key= lambda x: float(x.stem))
depth_names = [[x for x in Path(basepath / f'depth_{track}').iterdir() if x.suffix == '.pfm'] for track in depth_tracks]
depth_names = [sorted(track, key= lambda x: float(x.stem)) for track in depth_names]

print(f'found {len(rgb_names)} rgb images, {[len(track) for track in depth_names]} depth images in {basepath}')
assert len(rgb_names) > 0, "No image found"

# SAMPLE DATA

def sample_data(rate: float, names: list) -> list:
    samples = np.arange(0, len(names) - 1, floor(1 / rate))
    lower = samples[:-1]
    upper = samples[1:]
    mid = (lower + upper) // 2
    q1 = np.random.randint(lower + 1, mid).tolist()
    q2 = np.random.randint(mid + 1, upper).tolist()
    return sorted(random.sample(q1 + q2, (len(q1) + len(q2)) // 2))

if hparams.sample_rate < 1.0 - 1e-6:
    rgb_samples = sample_data(hparams.sample_rate, rgb_names)
    depth_samples = [sample_data(hparams.sample_rate, depth_names_track) for depth_names_track in depth_names]
else:
    rgb_samples = np.arange(0, len(rgb_names) - 1)
    depth_samples = [np.arange(0, len(depth_names_track) - 1) for depth_names_track in depth_names]

rgb_names = np.array(rgb_names)[rgb_samples]
depth_names = [np.array(depth_names[i])[depth_samples[i]] for i in range(len(depth_names))]
n_rgb, n_depth = len(rgb_names), [len(depth_names_track) for depth_names_track in depth_names]
n_samples = [n_rgb + n for n in n_depth]

# READ POSE

if hparams.pose_storage_format == 'trajectory':
    rgb_pose_raw = np.loadtxt(basepath / 'rgb_gt.txt').reshape(-1, 4, 4)[rgb_samples]
    depth_pose_raw = [np.loadtxt(basepath / f'depth_{depth_track}').reshape(-1, 4, 4)[depth_samples[i]] for i, depth_track in enumerate(depth_tracks)]
elif hparams.pose_storage_format == 'frame':
    def load_frame_wise_pose(pose_dir: Path, names: List[Path]):
        poses = np.zeros((len(names), 4, 4))
        for i, name in enumerate(names):
            pose_path = pose_dir / f'{name.stem}.txt'
            poses[i] = np.loadtxt(pose_path).reshape(4, 4)
        return poses
    rgb_pose_raw = load_frame_wise_pose(basepath / 'rgb', rgb_names)
    depth_pose_raw = [load_frame_wise_pose(basepath / f'depth_{track}', depth_names[i]) for i, track in enumerate(depth_tracks)]

# POSE PREPROCESS

def pose_transform(poses_raw):
    poses = []
    inv_pose = None
    c2b = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 0, -1]])#右下前变为下右后（相机坐标一般都是右下前）
    for b2w in poses_raw:
        c2w = b2w @ transformations.euler_matrix(0, -1.39626, 0) @ c2b
        if inv_pose is None:
            inv_pose = np.eye(4)
            inv_pose[:3, 3] = -c2w[:3, 3]
            c2w = inv_pose @ c2w
            c2w_RDF = np.linalg.inv(c2b) @ c2w
        else:
            c2w = inv_pose @ c2w
            c2w_RDF = np.linalg.inv(c2b) @ c2w
        c2w_RDF = torch.from_numpy(c2w_RDF).float()
        c2w_mega = torch.hstack((
            RDF_TO_DRB @ c2w_RDF[:3, :3] @ torch.inverse(RDF_TO_DRB),
            RDF_TO_DRB @ c2w_RDF[:3, 3:]
        ))
        poses += [c2w_mega]
    return poses

rgb_pose = pose_transform(rgb_pose_raw)
rgb_positions = torch.cat([c2w[:3, 3].unsqueeze(0) for c2w in rgb_pose])
depth_pose = [pose_transform(depth_pose_raw_track) for depth_pose_raw_track in depth_pose_raw]
depth_positions = [torch.cat([c2w[:3, 3].unsqueeze(0) for c2w in depth_pose_track]) for depth_pose_track in depth_pose]

max_values = rgb_positions.max(0)[0]
min_values = rgb_positions.min(0)[0]
origin = ((max_values + min_values) * 0.5)

# CHECK POSITION RANGE

def check_range(min_val, max_val):
    if (min_val < -hparams.pose_scale_factor).any() or (max_val > hparams.pose_scale_factor).any():
        warning('position out of [-1, 1] range')
check_range(min_values, max_values)

print(f"rgb poses ranges from {min_values} to {max_values}, centered at {origin}")
for i, depth_track in enumerate(depth_tracks):
    max_depth_position = depth_positions[i].max(0)[0]
    min_depth_position = depth_positions[i].min(0)[0]
    print(f"depth track {depth_track} ranges from {min_depth_position} to {max_depth_position}.")
    check_range(min_depth_position, max_depth_position)

# WRITE DATASET
pose_scale_factor = hparams.pose_scale_factor

dirs = [
    mega_path / 'train' / 'rgbs',
    mega_path / 'train' / 'metadata_rgb', 
    mega_path / 'val' / 'rgbs',
    mega_path / 'val' / 'metadata_rgb' ] + \
    [(mega_path / 'train' / f'depth_{track}') for track in depth_tracks] + \
    [(mega_path / 'train' / f'metadata_depth_{track}') for track in depth_tracks] + \
    [(mega_path / 'val' / f'depth_{track}') for track in depth_tracks] + \
    [(mega_path / 'val' / f'metadata_depth_{track}') for track in depth_tracks]

for dir in dirs:
    if not dir.exists():
        dir.mkdir(parents=True)

def save_track(name, poses, positions, image_names):
    with open(os.path.join(mega_path, f'mappings_{name}.txt'),mode='w') as f:
        for idx, _ in enumerate(tqdm(image_names)):
            if idx % int(positions.shape[0] / num_val) == 0:
                split_dir = os.path.join(mega_path,"val")
            else:
                split_dir = os.path.join(mega_path,"train")
            
            if name == 'rgb':
                color = cv2.imread(str(rgb_names[idx]),-1)
                cv2.imwrite(os.path.join(split_dir,'rgbs','{0:06d}.jpg'.format(idx)),color)
                image = color
            else:
                depth_path = image_names[idx]
                depth, scale = pfm_utils.read_pfm(depth_path)
                depth = np.flip(depth, axis=0)
                pfm_utils.write_pfm(os.path.join(split_dir, name, '{0:06d}.pfm'.format(idx)), depth, scale)
                image = depth

            camera_in_drb = poses[idx].clone() #这个操作会改变原来的
            camera_in_drb[:, 3] = (camera_in_drb[:, 3] - origin) / pose_scale_factor

            assert np.logical_and(camera_in_drb >= -1, camera_in_drb <= 1).all()
            metadata_name = '{0:06d}.pt'.format(idx)
            torch.save({
                'H': image.shape[0],
                'W': image.shape[1],
                'c2w' if name == 'rgb' else 'c2w_init': torch.cat(
                    [camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]],
                    -1), #变为右上后，又和nice-slam（nerf-pytorch保持一致了），局部为右上后，全局为下右后（不懂为啥要搞这么复杂）
                'intrinsics': torch.FloatTensor(
                    [K[0][0], K[1][1], K[0][2], K[1][2]]),
                'distortion': torch.FloatTensor(distortion),
                'timestamp': torch.tensor(float(image_names[idx].stem),dtype=torch.float64)
            }, os.path.join(split_dir,f'metadata_{name}',metadata_name))
            f.write('{},{}\n'.format(('{0:06d}.jpg' if name == 'rgb' else '{0:06d}.pfm').format(idx), metadata_name))

print("exporting rgb data")
save_track('rgb', rgb_pose, rgb_positions, rgb_names)
print("exporting depth data")
[save_track(f"depth_{track}", depth_pose[i], depth_positions[i], depth_names[i]) for i, track in enumerate(tqdm(depth_tracks))]

coordinates = {
    'origin_drb': origin,
    'pose_scale_factor': pose_scale_factor
}

torch.save(coordinates, os.path.join(mega_path,'coordinates.pt'))  # origin_drb, pose_scale_factor

print('DONE')