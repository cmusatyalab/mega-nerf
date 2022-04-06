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

from mega_nerf.datasets.dataset_utils import get_rgb_index_mask
from mega_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_ray_directions, get_rays, get_rays_batch

RAY_CHUNK_SIZE = 64 * 1024


class FilesystemDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, chunk_paths: List[Path], num_chunks: int,
                 scale_factor: int, disk_flush_size: int):
        super(FilesystemDataset, self).__init__()
        self._device = device
        self._c2ws = torch.cat([x.c2w.unsqueeze(0) for x in metadata_items])
        self._near = near
        self._far = far
        self._ray_altitude_range = ray_altitude_range

        intrinsics = torch.cat(
            [torch.cat([torch.FloatTensor([x.W, x.H]), x.intrinsics]).unsqueeze(0) for x in metadata_items])
        if (intrinsics - intrinsics[0]).abs().max() == 0:
            main_print(
                'All intrinsics identical: W: {} H: {}, intrinsics: {}'.format(metadata_items[0].W, metadata_items[0].H,
                                                                               metadata_items[0].intrinsics))

            self._directions = get_ray_directions(metadata_items[0].W,
                                                  metadata_items[0].H,
                                                  metadata_items[0].intrinsics[0],
                                                  metadata_items[0].intrinsics[1],
                                                  metadata_items[0].intrinsics[2],
                                                  metadata_items[0].intrinsics[3],
                                                  center_pixels,
                                                  device).view(-1, 3)
        else:
            main_print('Differing intrinsics')
            self._directions = None

        append_arrays = self._check_existing_paths(chunk_paths, center_pixels, scale_factor,
                                                   len(metadata_items))
        if append_arrays is not None:
            main_print('Reusing {} chunks from previous run'.format(len(append_arrays[0])))
            self._rgb_arrays = append_arrays[0]
            self._ray_arrays = append_arrays[1]
            self._img_arrays = append_arrays[2]
        else:
            self._rgb_arrays = []
            self._ray_arrays = []
            self._img_arrays = []
            self._write_chunks(metadata_items, center_pixels, device, chunk_paths, num_chunks, scale_factor,
                               disk_flush_size)

        self._rgb_arrays.sort(key=lambda x: x.name)
        self._ray_arrays.sort(key=lambda x: x.name)
        self._img_arrays.sort(key=lambda x: x.name)

        self._chunk_index = cycle(range(len(self._rgb_arrays)))
        self._loaded_rgbs = None
        self._loaded_rays = None
        self._loaded_image_indices = None
        self._chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_future = self._chunk_load_executor.submit(self._load_chunk_inner)
        self._chosen = None

    def load_chunk(self) -> None:
        chosen, self._loaded_rgbs, self._loaded_rays, self._loaded_image_indices = self._chunk_future.result()
        self._chosen = chosen
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

    def _load_chunk_inner(self) -> Tuple[
        str, torch.FloatTensor, torch.FloatTensor, torch.ShortTensor]:
        if 'RANK' in os.environ:
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

        next_index = next(self._chunk_index)
        chosen = self._rgb_arrays[next_index]
        loaded_img_indices = torch.ShortTensor(np.load(self._img_arrays[next_index]))

        if self._directions is not None:
            loaded_pixel_indices = torch.IntTensor(np.load(self._ray_arrays[next_index]))

            loaded_rays = []
            for i in range(0, loaded_pixel_indices.shape[0], RAY_CHUNK_SIZE):
                image_indices = loaded_img_indices[i:i + RAY_CHUNK_SIZE]
                unique_img_indices, inverse_img_indices = torch.unique(image_indices, return_inverse=True)
                c2ws = self._c2ws[unique_img_indices.long()].to(self._device)

                pixel_indices = loaded_pixel_indices[i:i + RAY_CHUNK_SIZE]
                unique_pixel_indices, inverse_pixel_indices = torch.unique(pixel_indices, return_inverse=True)

                # (#unique images, w*h, 8)
                image_rays = get_rays_batch(self._directions[unique_pixel_indices.long()],
                                            c2ws, self._near, self._far,
                                            self._ray_altitude_range).cpu()

                del c2ws

                loaded_rays.append(image_rays[inverse_img_indices, inverse_pixel_indices])

            loaded_rays = torch.cat(loaded_rays)
        else:
            loaded_rays = torch.FloatTensor(np.load(self._ray_arrays[next_index]))

        return str(chosen), torch.FloatTensor(np.load(chosen)) / 255., loaded_rays, loaded_img_indices

    def _write_chunks(self, metadata_items: List[ImageMetadata], center_pixels: bool, device: torch.device,
                      chunk_paths: List[Path], num_chunks: int, scale_factor: int, disk_flush_size: int) -> None:
        assert ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0

        path_frees = []
        total_free = 0

        for chunk_path in chunk_paths:
            (chunk_path / 'rgb-chunks').mkdir(parents=True)
            (chunk_path / 'ray-chunks').mkdir(parents=True)
            (chunk_path / 'img-chunks').mkdir(parents=True)

            _, _, free = shutil.disk_usage(chunk_path)
            total_free += free
            path_frees.append(free)

        rgb_append_arrays = []
        ray_append_arrays = []
        img_append_arrays = []

        index = 0
        for chunk_path, path_free in zip(chunk_paths, path_frees):
            allocated = int(path_free / total_free * num_chunks)
            main_print('Allocating {} chunks to dataset path {}'.format(allocated, chunk_path))
            for j in range(allocated):
                rgb_array_path = chunk_path / 'rgb-chunks' / '{}.npy'.format(index)
                self._rgb_arrays.append(rgb_array_path)
                rgb_append_arrays.append(NpyAppendArray(str(rgb_array_path)))

                ray_array_path = chunk_path / 'ray-chunks' / '{}.npy'.format(index)
                self._ray_arrays.append(ray_array_path)
                ray_append_arrays.append(NpyAppendArray(str(ray_array_path)))

                img_array_path = chunk_path / 'img-chunks' / '{}.npy'.format(index)
                self._img_arrays.append(img_array_path)
                img_append_arrays.append(NpyAppendArray(str(img_array_path)))
                index += 1
        main_print('{} chunks allocated'.format(index))

        write_futures = []
        rgbs = []
        rays = []
        indices = []
        in_memory_count = 0

        if self._directions is not None:
            all_pixel_indices = torch.arange(self._directions.shape[0], dtype=torch.int)

        with ThreadPoolExecutor(max_workers=len(rgb_append_arrays)) as executor:
            for metadata_item in main_tqdm(metadata_items):
                image_data = get_rgb_index_mask(metadata_item)

                if image_data is None:
                    continue

                image_rgbs, image_indices, image_keep_mask = image_data
                rgbs.append(image_rgbs)
                indices.append(image_indices)
                in_memory_count += len(image_rgbs)

                if self._directions is not None:
                    image_pixel_indices = all_pixel_indices
                    if image_keep_mask is not None:
                        image_pixel_indices = image_pixel_indices[image_keep_mask == True]

                    rays.append(image_pixel_indices)
                else:
                    directions = get_ray_directions(metadata_item.W,
                                                    metadata_item.H,
                                                    metadata_item.intrinsics[0],
                                                    metadata_item.intrinsics[1],
                                                    metadata_item.intrinsics[2],
                                                    metadata_item.intrinsics[3],
                                                    center_pixels,
                                                    device)
                    image_rays = get_rays(directions, metadata_item.c2w.to(device), self._near, self._far,
                                          self._ray_altitude_range).view(-1, 8).cpu()

                    if image_keep_mask is not None:
                        image_rays = image_rays[image_keep_mask == True]

                    rays.append(image_rays)

                if in_memory_count >= disk_flush_size:
                    for write_future in write_futures:
                        write_future.result()

                    write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                        rgb_append_arrays, ray_append_arrays, img_append_arrays)

                    rgbs = []
                    rays = []
                    indices = []
                    in_memory_count = 0

            for write_future in write_futures:
                write_future.result()

            if in_memory_count > 0:
                write_futures = self._write_to_disk(executor, torch.cat(rgbs), torch.cat(rays), torch.cat(indices),
                                                    rgb_append_arrays, ray_append_arrays, img_append_arrays)

                for write_future in write_futures:
                    write_future.result()
        for chunk_path in chunk_paths:
            chunk_metadata = {
                'images': len(metadata_items),
                'scale_factor': scale_factor
            }

            if self._directions is None:
                chunk_metadata['near'] = self._near
                chunk_metadata['far'] = self._far
                chunk_metadata['center_pixels'] = center_pixels
                chunk_metadata['ray_altitude_range'] = self._ray_altitude_range

            torch.save(chunk_metadata, chunk_path / 'metadata.pt')

        for source in [rgb_append_arrays, ray_append_arrays, img_append_arrays]:
            for append_array in source:
                append_array.close()

        main_print('Finished writing chunks to dataset paths')

    def _check_existing_paths(self, chunk_paths: List[Path], center_pixels: bool, scale_factor: int, images: int) -> \
            Optional[Tuple[List[Path], List[Path], List[Path]]]:
        rgb_arrays = []
        ray_arrays = []
        img_arrays = []

        num_exist = 0
        for chunk_path in chunk_paths:
            if chunk_path.exists():
                assert (chunk_path / 'metadata.pt').exists(), \
                    "Could not find metadata file (did previous writing to this directory not complete successfully?)"
                dataset_metadata = torch.load(chunk_path / 'metadata.pt', map_location='cpu')
                assert dataset_metadata['images'] == images
                assert dataset_metadata['scale_factor'] == scale_factor

                if self._directions is None:
                    assert dataset_metadata['near'] == self._near
                    assert dataset_metadata['far'] == self._far
                    assert dataset_metadata['center_pixels'] == center_pixels

                    if self._ray_altitude_range is not None:
                        assert (torch.allclose(torch.FloatTensor(dataset_metadata['ray_altitude_range']),
                                               torch.FloatTensor(self._ray_altitude_range)))
                    else:
                        assert dataset_metadata['ray_altitude_range'] is None

                for child in list((chunk_path / 'rgb-chunks').iterdir()):
                    rgb_arrays.append(child)
                    ray_arrays.append(child.parent.parent / 'ray-chunks' / child.name)
                    img_arrays.append(child.parent.parent / 'img-chunks' / child.name)
                num_exist += 1

        if num_exist > 0:
            assert num_exist == len(chunk_paths)
            return rgb_arrays, ray_arrays, img_arrays
        else:
            return None

    @staticmethod
    def _write_to_disk(executor: ThreadPoolExecutor, rgbs: torch.Tensor, rays: torch.FloatTensor,
                       image_indices: torch.Tensor, rgb_append_arrays, ray_append_arrays, img_append_arrays) -> List[
        Future[None]]:
        indices = torch.randperm(rgbs.shape[0])
        num_chunks = len(rgb_append_arrays)
        chunk_size = math.ceil(rgbs.shape[0] / num_chunks)

        futures = []

        def append(index: int) -> None:
            rgb_append_arrays[index].append(rgbs[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            ray_append_arrays[index].append(rays[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())
            img_append_arrays[index].append(image_indices[indices[index * chunk_size:(index + 1) * chunk_size]].numpy())

        for i in range(num_chunks):
            future = executor.submit(append, i)
            futures.append(future)

        return futures
