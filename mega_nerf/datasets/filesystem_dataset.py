import math
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from npy_append_array import NpyAppendArray
from torch.utils.data import Dataset

from mega_nerf.datasets.dataset_utils import get_image_data
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print


class FilesystemDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, chunk_paths: List[Path], num_chunks: int,
                 scale_factor: int, disk_flush_size: int):
        super(FilesystemDataset, self).__init__()

        append_arrays = self._check_existing_paths(chunk_paths, near, far, ray_altitude_range, center_pixels,
                                                   scale_factor, len(metadata_items))
        if append_arrays is not None:
            main_print('Reusing {} chunks from previous run'.format(len(append_arrays)))
            self._append_arrays = append_arrays
        else:
            self._append_arrays = []
            self._write_chunks(metadata_items, near, far, ray_altitude_range, center_pixels, device, chunk_paths,
                               num_chunks, scale_factor, disk_flush_size)

        self._append_arrays.sort(key=lambda x: x.filename)
        self._chunk_index = cycle(range(len(self._append_arrays)))
        self._loaded_rgbs = None
        self._loaded_rays = None
        self._loaded_image_indices = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self._chosen = None

    def load_chunk(self) -> None:
        chosen, loaded_chunk = self._chunk_future.result()
        self._chosen = chosen
        self._loaded_rgbs = loaded_chunk[:, :3]
        self._loaded_rays = loaded_chunk[:, 3:11]
        self._loaded_image_indices = loaded_chunk[:, 11]
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)

    def get_state(self) -> str:
        return self._chosen

    def set_state(self, chosen: str) -> None:
        while self._chosen != chosen:
            self.load_chunk()

    def __len__(self) -> int:
        return self._loaded_rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'rgbs': self._loaded_rgbs[idx],
            'rays': self._loaded_rays[idx],
            'image_indices': self._loaded_image_indices[idx]
        }

    def _load_chunk_inner(self) -> Tuple[str, torch.FloatTensor]:
        chosen = self._append_arrays[next(self._chunk_index)]
        return str(chosen.filename), torch.FloatTensor(np.load(chosen.filename))

    def _write_chunks(self, metadata_items: List[ImageMetadata], near: float, far: float,
                      ray_altitude_range: List[float], center_pixels: bool, device: torch.device,
                      chunk_paths: List[Path], num_chunks: int, scale_factor: int, disk_flush_size: int) -> None:
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        path_frees = []
        total_free = 0

        for chunk_path in chunk_paths:
            (chunk_path / 'chunks').mkdir(parents=True)
            _, _, free = shutil.disk_usage(chunk_path)
            total_free += free
            path_frees.append(free)

        index = 0
        for chunk_path, path_free in zip(chunk_paths, path_frees):
            allocated = int(path_free / total_free * num_chunks)
            main_print('Allocating {} chunks to dataset path {}'.format(allocated, chunk_path))
            for j in range(allocated):
                self._append_arrays.append(NpyAppendArray(str(chunk_path / 'chunks' / '{}.npy'.format(index))))
                index += 1
        main_print('{} chunks allocated'.format(index))

        write_futures = []
        rgbs = []
        rays = []
        indices = []
        in_memory_count = 0
        with ThreadPoolExecutor(max_workers=len(self._append_arrays)) as executor:
            for metadata_item in main_tqdm(metadata_items):
                image_data = get_image_data(metadata_item, near, far, ray_altitude_range, center_pixels, device)

                if image_data is None:
                    continue

                image_rgbs, image_rays, image_indices = image_data

                rgbs.append(image_rgbs)
                rays.append(image_rays)
                indices.append(image_indices)
                in_memory_count += len(image_rgbs)

                if in_memory_count >= disk_flush_size:
                    for write_future in write_futures:
                        write_future.result()

                    write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices))

                    rgbs = []
                    rays = []
                    indices = []
                    in_memory_count = 0

            for write_future in write_futures:
                write_future.result()

            if in_memory_count > 0:
                write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices))

                for write_future in write_futures:
                    write_future.result()
        for chunk_path in chunk_paths:
            torch.save({
                'images': len(metadata_items),
                'scale_factor': scale_factor,
                'near': near,
                'far': far,
                'center_pixels': center_pixels,
                'ray_altitude_range': ray_altitude_range,
            }, chunk_path / 'metadata.pt')

        main_print('Finished writing chunks to dataset paths')

    def _check_existing_paths(self, chunk_paths: List[Path], near: float, far: float, ray_altitude_range: List[float],
                              center_pixels: bool, scale_factor: int, images: int) -> Optional[List[NpyAppendArray]]:
        append_arrays = []
        num_exist = 0
        for chunk_path in chunk_paths:
            if chunk_path.exists():
                dataset_metadata = torch.load(chunk_path / 'metadata.pt', map_location='cpu')
                assert dataset_metadata['images'] == images
                assert dataset_metadata['scale_factor'] == scale_factor
                assert dataset_metadata['near'] == near
                assert dataset_metadata['far'] == far
                assert dataset_metadata['center_pixels'] == center_pixels

                if ray_altitude_range is not None:
                    assert (torch.allclose(torch.FloatTensor(dataset_metadata['ray_altitude_range']),
                                           torch.FloatTensor(ray_altitude_range)))
                else:
                    assert dataset_metadata['ray_altitude_range'] is None

                for child in list((chunk_path / 'chunks').iterdir()):
                    append_arrays.append(NpyAppendArray(child))
                num_exist += 1

        if num_exist > 0:
            assert num_exist == len(chunk_paths)
            return append_arrays
        else:
            return None

    def _write_to_disk(self, executor: ThreadPoolExecutor, rgbs: torch.Tensor, rays: torch.Tensor,
                       image_indices: torch.Tensor) -> List[Future[None]]:
        to_store = torch.cat([rgbs, rays, image_indices.unsqueeze(-1)], -1)
        indices = torch.randperm(to_store.shape[0])
        num_chunks = len(self._append_arrays)
        chunk_size = math.ceil(to_store.shape[0] / num_chunks)

        futures = []

        def append(index: int) -> None:
            self._append_arrays[index].append(
                to_store[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())

        for i in range(num_chunks):
            future = executor.submit(append, i)
            futures.append(future)

        return futures
