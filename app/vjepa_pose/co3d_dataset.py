#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gzip
import json
import os
import random
from collections import defaultdict
from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch

from app.vjepa_pose.pose_dataset import PoseVideoDataset

logger = getLogger(__name__)


def _load_jgz(path: str):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _co3d_viewpoint_to_c2w(viewpoint: dict) -> torch.Tensor:
    """Convert a CO3D / PyTorch3D viewpoint to a 4x4 camera-to-world matrix.

    PyTorch3D right-multiply convention (row vectors)::

        X_cam = X_world @ R + T

    Camera-to-world in column-vector convention::

        c2w = [[R,   -R @ T],
               [0 0 0,    1]]
    """
    R = np.array(viewpoint["R"], dtype=np.float32)   # (3, 3)
    T = np.array(viewpoint["T"], dtype=np.float32)   # (3,)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = -R @ T
    return torch.from_numpy(c2w)


class Co3dDataset(PoseVideoDataset):
    """Pose dataset for CO3D (Common Objects in 3D).

    Expected directory layout::

        data_path/
            <category>/
                frame_annotations.jgz
                sequence_annotations.jgz
                splits.json              (created by create_splits.py)
                <sequence_name>/
                    images/
                        frame000001.jpg
                        ...

    Parameters
    ----------
    data_path : str
        Root of the CO3D download (parent of category directories).
    category : str
        Object category, e.g. ``"teddybear"``.
    split : str or None
        ``"train"``, ``"val"``, or ``None`` (use all sequences).
    """

    def __init__(
        self,
        data_path: str,
        category: str = "teddybear",
        split: str = "train",
        frames_per_clip: int = 16,
        transform=None,
        sequential: bool = True,
        **kwargs,
    ):
        super().__init__(data_path, frames_per_clip=frames_per_clip,
                         transform=transform)
        self.category = category
        self.split = split
        self.sequential = sequential

        category_dir = os.path.join(self.data_path, category)
        if not os.path.isdir(category_dir):
            raise NotADirectoryError(
                f"Category directory not found: {category_dir}"
            )

        # --- load frame annotations -----------------------------------------
        frame_annotations = _load_jgz(
            os.path.join(category_dir, "frame_annotations.jgz")
        )

        seq_map: dict[str, list] = defaultdict(list)
        for fa in frame_annotations:
            seq_map[fa["sequence_name"]].append(fa)
        for frames in seq_map.values():
            frames.sort(key=lambda x: x["frame_number"])

        # --- filter by split -------------------------------------------------
        if split is not None:
            splits_path = os.path.join(category_dir, "splits.json")
            if not os.path.isfile(splits_path):
                raise FileNotFoundError(
                    f"Split file not found: {splits_path}.  "
                    "Run  third_party/co3d/create_splits.py  first."
                )
            with open(splits_path, "r") as f:
                splits = json.load(f)
            if split not in splits:
                raise KeyError(
                    f"Split {split!r} not in {splits_path} "
                    f"(available: {list(splits.keys())})"
                )
            allowed = set(splits[split])
            seq_map = {k: v for k, v in seq_map.items() if k in allowed}

        if not seq_map:
            raise RuntimeError(
                f"No sequences for split={split!r} in {category_dir}"
            )

        # --- build sequence list (one sample = one sequence) -----------------
        self._sequences: list[dict] = []

        for seq_name in sorted(seq_map):
            frames = seq_map[seq_name]
            if len(frames) < self.frames_per_clip:
                continue

            img_paths = [
                os.path.join(self.data_path, fa["image"]["path"])
                for fa in frames
            ]
            poses = [_co3d_viewpoint_to_c2w(fa["viewpoint"]) for fa in frames]

            self._sequences.append({
                "name": seq_name,
                "image_paths": img_paths,
                "poses": poses,
            })

        logger.info(
            "Co3dDataset: category=%s  split=%s  sequences=%d",
            category, split, len(self._sequences),
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_clip_paths_and_poses(
        self, index: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        if index < 0 or index >= len(self._sequences):
            raise IndexError(
                f"Index {index} out of range for {len(self._sequences)} sequences."
            )

        seq = self._sequences[index]
        n_frames = len(seq["image_paths"])

        if self.sequential:
            start = random.randint(0, n_frames - self.frames_per_clip)
            indices = list(range(start, start + self.frames_per_clip))
        else:
            indices = sorted(random.sample(range(n_frames), self.frames_per_clip))

        paths = [seq["image_paths"][i] for i in indices]
        poses = [seq["poses"][i] for i in indices]
        return paths, poses

    def __len__(self) -> int:
        return len(self._sequences)
