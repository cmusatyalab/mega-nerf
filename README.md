# Mega-NeRF

This repository contains the code needed to train Mega-NeRF models and generate the sparse voxel octrees used by the Mega-NeRF-Dynamic viewer.

The codebase for the Mega-NeRF-Dynamic viewer can be found [here](https://github.com/cmusatyalab/mega-nerf-viewer).

**Note:** This is a preliminary release and there may still be outstanding bugs.

## Demo
![](demo/rubble-orbit.gif)
![](demo/building-orbit.gif)

## Data

### Mill 19

The two Mill 19 scenes can be downloaded at: TODO

### UrbanScene 3D

1. Download the raw photo collections from the [UrbanScene3D](https://vcc.tech/UrbanScene3D/) dataset
2. Download the refined camera poses from TODO
3. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Quad 6k Dataset

1. Download the raw photo collections from [here](http://vision.soic.indiana.edu/disco_files/ArtsQuad_dataset.tar).
2. Download the refined camera poses from TODO
3. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Custom Data

The expected directory structure is:
- /coordinates.pt: [Torch file](https://pytorch.org/docs/stable/generated/torch.save.html) that should contain the following keys:
  - 'origin_drb': Origin of scene in real-world units
  - 'pose_scale_factor': Scale factor mapping from real-world unit (ie: meters) to [-1, 1] range
- '/{val|train}/rgbs/': JPEG or PNG images
- '/{val|train}/metadata/': Image-specific image metadata saved as a torch file. Each image should have a corresponding metadata file with the following file format: {rgb_stem}.pt. Each metadata file should contain the following keys:
-- 'W': Image width
-- 'H': Image height
-- 'intrinsics': Image intrinsics in the following form: [fx, fy, cx, cy]
-- 'c2w': Camera pose. 3x3 camera matrix with the convention used in the original [NeRF repo](https://github.com/bmild/nerf), ie: x: down, y: right, z: backwards, followed by the following transformation: ```torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1)```

## Training

### Generating training partitions

