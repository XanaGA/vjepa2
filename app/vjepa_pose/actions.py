#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Action computation from camera pose sequences.

An action function has the signature::

    (pose_matrix: Tensor[T, 4, 4]) -> Tensor[T-1, D]

where *D* is the action dimensionality (5 for the default).
"""

import math
from typing import Callable

import torch

from app.vjepa_pose.pose_dataset import c2w_to_pose5

ActionFn = Callable[[torch.Tensor], torch.Tensor]


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [−π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def finite_difference_actions(pose_matrix: torch.Tensor) -> torch.Tensor:
    """Compute actions as finite differences on the 5-number pose.

    Parameters
    ----------
    pose_matrix : Tensor [T, 4, 4]

    Returns
    -------
    actions : Tensor [T-1, 5]
        ``(dx, dy, dz, dtheta, dphi)`` — translation delta and
        angular deltas between consecutive frames, with angles
        wrapped to [−π, π].
    """
    p5 = c2w_to_pose5(pose_matrix)      # [T, 5]
    delta = p5[1:] - p5[:-1]            # [T-1, 5]
    delta[..., 3] = _wrap_angle(delta[..., 3])
    delta[..., 4] = _wrap_angle(delta[..., 4])
    return delta
