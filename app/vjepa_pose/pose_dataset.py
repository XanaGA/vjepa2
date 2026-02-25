#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from abc import ABC, abstractmethod
from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image

_GLOBAL_SEED = 0
logger = getLogger(__name__)


# ======================================================================
# Pose conversions: 4x4 c2w matrix  ⟷  5-number (x, y, z, θ, φ)
# ======================================================================

def c2w_to_pose5(c2w: torch.Tensor) -> torch.Tensor:
    """Convert camera-to-world matrices to a compact 5-number pose.

    Parameters
    ----------
    c2w : Tensor [*, 4, 4]

    Returns
    -------
    pose5 : Tensor [*, 5]
        ``(x, y, z, theta, phi)`` where ``(x, y, z)`` is the camera
        position and ``(theta, phi)`` are the spherical-coordinate angles
        of the camera Z-axis (look direction) in world frame.

        * ``theta`` ∈ [0, π]  — polar angle from world +Z.
        * ``phi``   ∈ [−π, π] — azimuthal angle in the XY plane.
    """
    pos = c2w[..., :3, 3]                                   # [*, 3]
    look = c2w[..., :3, 2]                                  # [*, 3]
    theta = torch.acos(look[..., 2].clamp(-1.0, 1.0))       # [*]
    phi = torch.atan2(look[..., 1], look[..., 0])            # [*]
    return torch.cat([pos, theta.unsqueeze(-1),
                      phi.unsqueeze(-1)], dim=-1)            # [*, 5]


def pose5_to_c2w(pose5: torch.Tensor) -> torch.Tensor:
    """Reconstruct camera-to-world matrices from the 5-number pose.

    Roll is fixed to zero (camera Y as close to world +Y as possible).

    Parameters
    ----------
    pose5 : Tensor [*, 5]

    Returns
    -------
    c2w : Tensor [*, 4, 4]
    """
    pos = pose5[..., :3]
    theta = pose5[..., 3]
    phi = pose5[..., 4]

    sin_t = torch.sin(theta)
    z_axis = torch.stack([sin_t * torch.cos(phi),
                          sin_t * torch.sin(phi),
                          torch.cos(theta)], dim=-1)         # [*, 3]

    # Gram-Schmidt: x = up × z, y = z × x (zero-roll convention)
    up = torch.zeros_like(z_axis)
    up[..., 1] = 1.0                                        # world +Y

    x_axis = torch.linalg.cross(up, z_axis)
    x_norm = x_axis.norm(dim=-1, keepdim=True)

    # Fallback when look ≈ ±Y (cross product degenerates)
    fallback = torch.zeros_like(z_axis)
    fallback[..., 2] = 1.0                                  # world +Z
    x_fallback = torch.linalg.cross(fallback, z_axis)

    degenerate = (x_norm < 1e-6).expand_as(x_axis)
    x_axis = torch.where(degenerate, x_fallback, x_axis)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    y_axis = torch.linalg.cross(z_axis, x_axis)

    batch_shape = pose5.shape[:-1]
    c2w = torch.zeros(*batch_shape, 4, 4,
                       dtype=pose5.dtype, device=pose5.device)
    c2w[..., :3, 0] = x_axis
    c2w[..., :3, 1] = y_axis
    c2w[..., :3, 2] = z_axis
    c2w[..., :3, 3] = pos
    c2w[..., 3, 3] = 1.0
    return c2w


class PoseVideoDataset(ABC, torch.utils.data.Dataset):
    """Abstract base class for pose-video datasets.

    Each sample is a temporal clip of ``frames_per_clip`` consecutive frames.

    Returns
    -------
    dict with keys
        ``"buffer"``      : Tensor [C, T, H, W]
        ``"pose_matrix"`` : Tensor [T, 4, 4]  — full camera-to-world matrix
        ``"pose5"``       : Tensor [T, 5]      — (x, y, z, θ, φ)
    """

    def __init__(self, data_path: str, frames_per_clip: int = 16,
                 transform=None, **kwargs):
        super().__init__()
        self.data_path = os.path.expanduser(data_path)
        self.frames_per_clip = int(frames_per_clip)
        self.transform = transform

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_clip_paths_and_poses(
        self, index: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """Return ``(image_paths, poses)`` for clip *index*.

        ``image_paths`` is a list of length ``frames_per_clip``.
        ``poses`` is a list of ``[4, 4]`` float32 tensors.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def __getitem__(self, index: int):
        image_paths, poses_list = self._get_clip_paths_and_poses(index)

        frames = []
        target_size = None  # (W, H) of the first frame — used to homogenise
        for img_path in image_paths:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            if target_size is None:
                target_size = img.size          # (W, H)
            elif img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            frames.append(np.array(img, dtype=np.uint8))

        buffer = np.stack(frames, axis=0)  # [T, H, W, C]

        if self.transform is not None:
            buffer = self.transform(buffer)  # expected [C, T, H, W]
        else:
            buffer = torch.from_numpy(buffer)        # [T, H, W, C]
            buffer = buffer.permute(3, 0, 1, 2)      # [C, T, H, W]

        pose_matrix = torch.stack(poses_list, dim=0)  # [T, 4, 4]
        pose5 = c2w_to_pose5(pose_matrix)              # [T, 5]

        return {
            "buffer": buffer,
            "pose_matrix": pose_matrix,
            "pose5": pose5,
        }


# ======================================================================
# Factory / DataLoader helper
# ======================================================================

def init_data(
    data_path,
    batch_size,
    dataset_type="nerf",
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=0,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
    **dataset_kwargs,
):
    """Create a pose-video DataLoader.

    Parameters
    ----------
    dataset_type : str
        ``"nerf"`` for NeRF-synthetic scenes or ``"co3d"`` for CO3D.
    dataset_kwargs
        Extra keyword arguments forwarded to the concrete dataset class
        (e.g. *category* and *split* for CO3D).
    """
    if dataset_type == "nerf":
        from app.vjepa_pose.nerf_dataset import NerfDataset
        dataset = NerfDataset(
            data_path=data_path,
            frames_per_clip=frames_per_clip,
            transform=transform,
        )
    elif dataset_type == "co3d":
        from app.vjepa_pose.co3d_dataset import Co3dDataset
        dataset = Co3dDataset(
            data_path=data_path,
            frames_per_clip=frames_per_clip,
            transform=transform,
            **dataset_kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("PoseVideoDataset data loader created")
    return data_loader, dist_sampler
