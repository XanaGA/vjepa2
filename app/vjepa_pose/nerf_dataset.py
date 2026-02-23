#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from typing import List, Tuple

import torch

from app.vjepa_pose.pose_dataset import PoseVideoDataset


def _load_json(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


class NerfDataset(PoseVideoDataset):
    """Pose dataset for NeRF synthetic scenes.

    Expected directory layout::

        data_path/
            transforms_train.json
            train/
                r_0.png
                r_1.png
                ...
    """

    def __init__(
        self,
        data_path: str,
        frames_per_clip: int = 16,
        transform=None,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(data_path, frames_per_clip=frames_per_clip,
                         transform=transform)

        json_path = os.path.join(self.data_path, f"transforms_{split}.json")
        meta = _load_json(json_path)

        self.camera_angle_x = meta.get("camera_angle_x")
        self.image_paths: List[str] = []
        self.poses: List[torch.Tensor] = []

        for frame in meta.get("frames", []):
            rel_path = frame["file_path"]
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            self.image_paths.append(
                os.path.join(self.data_path, rel_path + ".png")
            )
            self.poses.append(
                torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            )

        if not self.image_paths:
            raise RuntimeError(f"No frames found in {json_path}")

        self._num_clips = max(
            0, len(self.image_paths) - self.frames_per_clip + 1
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_clip_paths_and_poses(
        self, index: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        if index < 0 or index >= self._num_clips:
            raise IndexError(
                f"Index {index} out of range for {self._num_clips} clips."
            )
        end = index + self.frames_per_clip
        return self.image_paths[index:end], self.poses[index:end]

    def __len__(self) -> int:
        return self._num_clips
