import random
import glob
import os
import rasterio
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class GalaxEyeDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), augment=False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        self.pre_paths = sorted(glob.glob(os.path.join(root_dir, "**/pre-event/*.tif"), recursive=True))
        self.post_paths = sorted(glob.glob(os.path.join(root_dir, "**/post-event/*.tif"), recursive=True))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "**/target/*.tif"), recursive=True))

    def __len__(self):
        return len(self.pre_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.pre_paths[idx]) as s: pre = s.read().transpose(1, 2, 0)
        with rasterio.open(self.post_paths[idx]) as s: post = s.read().transpose(1, 2, 0)
        with rasterio.open(self.mask_paths[idx]) as s: mask = s.read(1)

        pre = cv2.resize(pre, self.img_size)
        post = cv2.resize(post, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Label Remapping
        binary_mask = np.where((mask == 2) | (mask == 3), 1, 0).astype(np.float32)

        # Data Augmentation
        if self.augment:
            if random.random() > 0.5:
                pre, post, binary_mask = np.fliplr(pre).copy(), np.fliplr(post).copy(), np.fliplr(binary_mask).copy()
            if random.random() > 0.5:
                pre, post, binary_mask = np.flipud(pre).copy(), np.flipud(post).copy(), np.flipud(binary_mask).copy()

        pre = torch.from_numpy(pre).permute(2, 0, 1).float() / 255.0
        if len(post.shape) == 2: post = np.expand_dims(post, axis=-1)
        post = torch.from_numpy(post).permute(2, 0, 1).float() / 255.0

        return pre, post, torch.from_numpy(binary_mask).unsqueeze(0)
