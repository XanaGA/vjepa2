#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger

import numpy as np
import torch
import torch.utils.data
from PIL import Image

_GLOBAL_SEED = 0
logger = getLogger(__name__)


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,  # unused for now
    crop_size=224,  # unused here; augmentations handle sizing
    rank=0,
    world_size=1,
    camera_views=0,  # unused for this dataset type
    stereo_view=False,  # unused for this dataset type
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,
    tubelet_size=2,
):
    """
    Initialize pose dataset data loader.

    Each sample is a temporal clip of length `frames_per_clip`, formed by
    consecutive frames (images) and their corresponding poses.
    """
    dataset = PoseVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
    )

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


def get_json(path):
    """
    Load JSON metadata from a file.

    Args:
        path (str): Path to a JSON file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


class PoseVideoDataset(torch.utils.data.Dataset):
    """
    Pose dataset for NeRF synthetic data.

    Each sample corresponds to a single RGB image and its 4x4 camera
    pose matrix (both indexed from `transforms_<split>.json`).

    The expected directory structure (for `split="train"`) is:
        data_path/
            transforms_train.json
            train/
                r_0.png
                r_1.png
                ...
    """

    def __init__(
        self,
        data_path,
        camera_views=["left_mp4_path", "right_mp4_path"],
        frameskip=2,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
    ):
        """
        Args:
            data_path (str): Path to the NeRF synthetic scene root, e.g.
                '~/Downloads/nerf_synthetic/lego'.
            frames_per_clip (int): Number of consecutive frames in each sample.
            transform (callable, optional): Transform to apply to a
                buffer with shape [T, H, W, C] (uint8), where T is the
                number of frames (T == frames_per_clip).
        """
        super().__init__()

        self.data_path = os.path.expanduser(data_path)
        self.transform = transform
        self.frames_per_clip = int(frames_per_clip)

        # For now we hard-code the split to "train". This can be made
        # configurable later if needed.
        split = "train"
        json_path = os.path.join(self.data_path, f"transforms_{split}.json")
        meta = get_json(json_path)

        self.camera_angle_x = meta.get("camera_angle_x", None)

        self.image_paths = []
        self.poses = []

        for frame in meta.get("frames", []):
            # file_path is like "./train/r_0"
            rel_path = frame["file_path"]
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            img_path = os.path.join(self.data_path, rel_path + ".png")

            self.image_paths.append(img_path)
            # transform_matrix is 4x4
            pose_matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            self.poses.append(pose_matrix)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No frames found in metadata file: {json_path}")

        # Number of valid temporal clips of length `frames_per_clip`.
        # We only create clips that are fully inside the sequence.
        self.num_clips = max(0, len(self.image_paths) - self.frames_per_clip + 1)

    def __getitem__(self, index):
        """
        Returns:
            buffer (Tensor): Video-like tensor of shape [C, T, H, W]
                with T == frames_per_clip.
            poses (Tensor): Pose tensor of shape [T, 4, 4],
                containing the 4x4 camera transform matrices for each frame.
        """
        if index < 0 or index >= self.num_clips:
            raise IndexError(f"Index {index} out of range for {self.num_clips} clips.")

        start = index
        end = start + self.frames_per_clip

        frames = []
        poses_list = []
        for i in range(start, end):
            img_path = self.image_paths[i]
            pose = self.poses[i]

            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # Load image as H x W x C uint8
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.uint8)  # [H, W, C]
            frames.append(img_np)
            poses_list.append(pose)

        # Stack into [T, H, W, C]
        buffer = np.stack(frames, axis=0)  # [T, H, W, C]

        if self.transform is not None:
            buffer = self.transform(buffer)  # expected [C, T, H, W]
        else:
            # Default: convert to [C, T, H, W]
            buffer = torch.from_numpy(buffer)  # [T, H, W, C]
            buffer = buffer.permute(3, 0, 1, 2)  # [C, T, H, W]

        # Pose per frame, shape [T, 4, 4]
        poses = torch.stack(poses_list, dim=0)  # [T, 4, 4]

        return buffer, poses

    def __len__(self):
        return self.num_clips

