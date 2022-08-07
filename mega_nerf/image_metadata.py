from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: Path, depth_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool):
        self.image_path = image_path
        self.depth_path = depth_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val

    def load_image(self) -> torch.Tensor:
        """
        从文件系统中读取图片, 并按照 metadata 进行缩放
        Returns:
        -  torch.Tensor: 图片的缩放后的 tensor (self.W, self.H, 3)
        """
        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))
    
    def load_depth_image(self) -> torch.Tensor:
        """
        从文件系统中读取深度图片, 并按照 metadata 进行缩放
        Returns:
        - torch.Tensor: 深度图片的缩放后的 tensor (self.W, self.H, 1)
        """
        depths = Image.open(self.depth_path).convert('L')
        size = depths.size
        
        if size[0] != self.W or size[1] != self.H:
            depths = depths.resize((self.W, self.H), Image.LANCZOS)
        
        # 不知道欸什么读入的深度图是倒过来的, 这里做一下处理
        return torch.FloatTensor(np.asarray(depths)).flip(0) / 255

    def load_mask(self) -> Optional[torch.Tensor]:
        """
        加载数据集 mask, 若没有指定 mask, 返回 None
        Returns:
        - (Optional) torch.Tensor: mask tensor (self.W, self.H)
            在当前数据集中的像素 mask 值为 True, 否则为 False
        """
        if self._mask_path is None:
            return None

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
