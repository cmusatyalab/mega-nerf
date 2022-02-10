from typing import List, Tuple, Optional

import torch

from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.ray_utils import get_ray_directions, get_rays


def get_image_data(metadata: ImageMetadata, near: float, far: float, ray_altitude_range: List[float],
                   center_pixels: bool, device: torch.device) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    directions = get_ray_directions(metadata.W,
                                    metadata.H,
                                    metadata.intrinsics[0],
                                    metadata.intrinsics[1],
                                    metadata.intrinsics[2],
                                    metadata.intrinsics[3],
                                    center_pixels,
                                    device)
    rgbs = metadata.load_image().view(-1, 3)
    rays = get_rays(directions, metadata.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()

    keep_mask = metadata.load_mask()

    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:
            return None

        keep_mask = keep_mask.view(-1)

        rays = rays[keep_mask == True]
        rgbs = rgbs[keep_mask == True]

    return rgbs, rays, metadata.image_index * torch.ones(rgbs.shape[0])
