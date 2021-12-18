from typing import List, Dict

import torch
from torch.utils.data import Dataset

from mega_nerf.datasets.dataset_utils import get_image_data
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print


class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device):
        super(MemoryDataset, self).__init__()

        rgbs = []
        rays = []
        indices = []

        main_print('Loading data')

        for metadata_item in main_tqdm(metadata_items):
            image_data = get_image_data(metadata_item, near, far, ray_altitude_range, center_pixels, device)

            if image_data is None:
                continue

            image_rgbs, image_rays, image_indices = image_data

            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)

        main_print('Finished loading data')

        self._rgbs = torch.cat(rgbs)
        self._rays = torch.cat(rays)
        self._image_indices = torch.cat(indices)


    def __len__(self) -> int:
        return self._rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._rgbs[idx],
            'rays': self._rays[idx],
            'image_indices': self._image_indices[idx]
        }
