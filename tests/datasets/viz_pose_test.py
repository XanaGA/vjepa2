#!/usr/bin/env python3
#
# Utility script to visualize PoseVideoDataset samples:
#  - RGB images for a subset of samples
#  - Corresponding camera poses in 3D using Open3D
#
# Supports both NeRF-synthetic and CO3D datasets.
#
# Examples
# --------
# NeRF:
#   python tests/datasets/viz_pose_test.py \
#       --dataset-type nerf \
#       --data-path ~/Downloads/nerf_synthetic/lego
#
# CO3D:
#   python tests/datasets/viz_pose_test.py \
#       --dataset-type co3d \
#       --data-path data/co3d \
#       --category teddybear \
#       --split train

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch


def create_camera_mesh(scale=0.1, color=(0.9, 0.1, 0.1)):
    """
    Create a simple wireframe camera frustum as an Open3D LineSet.
    The canonical camera looks along +Z with its center at the origin.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # camera center
            [1.0, 1.0, 2.0],  # top-right
            [-1.0, 1.0, 2.0],  # top-left
            [-1.0, -1.0, 2.0],  # bottom-left
            [1.0, -1.0, 2.0],  # bottom-right
        ]
    )
    vertices *= float(scale)

    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],  # center to corners
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],  # base rectangle
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color, dtype=np.float32)[None, :], (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_sample_images(dataset, sample_id, max_frames=8):
    """
    Visualize up to `max_frames` RGB images for a single logical sample,
    laid out in a grid in a single matplotlib figure.
    """
    n = max_frames
    if n == 0:
        return

    cols = min(n, max_frames)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    buffer, poses = dataset[sample_id]  # buffer: [C, T, H, W]
    buffer = buffer.permute(1, 0, 2, 3)  # -> [T, C, H, W]

    for ax, idx in zip(axes, range(len(buffer))):
        img = buffer[idx].permute(1, 2, 0)  # [H, W, C]
        img_np = img.cpu().numpy()

        if img_np.dtype != np.uint8:
            img_min = img_np.min()
            img_max = img_np.max()
            if img_max > img_min:
                img_np = (255.0 * (img_np - img_min) / (img_max - img_min)).astype(np.uint8)
            else:
                img_np = np.zeros_like(img_np, dtype=np.uint8)

        ax.imshow(img_np)
        ax.set_title(f"idx {idx}")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Sample {sample_id}: first {n} frames")
    plt.tight_layout()
    plt.show()


def visualize_sample_poses(dataset, sample_id, negate_rotation=False):
    """
    Visualize camera poses in 3D for a single logical sample using Open3D.

    Both NeRF and CO3D datasets return 4x4 camera-to-world matrices, but
    the camera look direction differs:

    - NeRF (OpenGL): camera looks along -Z → set *negate_rotation=True*
      so the +Z frustum is flipped to match.
    - CO3D (PyTorch3D): camera looks along +Z → *negate_rotation=False*.
    """
    geometries = []

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    _, poses = dataset[sample_id]  # [T, 4, 4]

    cam_positions = poses[:, :3, 3].numpy()
    median_dist = np.median(np.linalg.norm(cam_positions, axis=1))
    frustum_scale = max(0.02, median_dist * 0.06)

    n_frames = len(poses)
    for idx in range(n_frames):
        pose_np = poses[idx].cpu().numpy()
        R = pose_np[:3, :3]
        t = pose_np[:3, 3]

        frac = idx / max(n_frames - 1, 1)
        color = (frac, 0.2, 1.0 - frac)

        camera_mesh = create_camera_mesh(scale=frustum_scale, color=color)
        camera_mesh.translate(t)
        rot = -R if negate_rotation else R
        camera_mesh.rotate(rot, center=t)

        geometries.append(camera_mesh)

    if len(geometries) > 1:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Poses – sample {sample_id}",
            width=1280,
            height=720,
        )


def build_dataset(args):
    """Construct the appropriate dataset from CLI arguments."""
    data_path = os.path.expanduser(args.data_path)
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"Invalid data path: {data_path}")

    if args.dataset_type == "nerf":
        from app.vjepa_pose.nerf_dataset import NerfDataset
        return NerfDataset(
            data_path=data_path,
            transform=None,
            frames_per_clip=args.frames_per_sample,
        )

    if args.dataset_type == "co3d":
        from app.vjepa_pose.co3d_dataset import Co3dDataset
        return Co3dDataset(
            data_path=data_path,
            category=args.category,
            split=args.split,
            transform=None,
            frames_per_clip=args.frames_per_sample,
            sequential=not args.random_order,
        )

    raise ValueError(f"Unknown dataset type: {args.dataset_type}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize PoseVideoDataset samples: show images with matplotlib "
            "and camera poses in 3D with Open3D."
        )
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["nerf", "co3d"],
        default="nerf",
        help="Dataset type to visualize (default: nerf).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="For nerf: scene root.  For co3d: dataset root (parent of category dirs).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="teddybear",
        help="[co3d only] Object category.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="[co3d only] Split to load: train, val, or omit for all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of logical samples to visualize.",
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        default=8,
        help="Number of consecutive frames per clip.",
    )
    parser.add_argument(
        "--random-order",
        action="store_true",
        help="Load frames in random order instead of sequentially.",
    )
    args = parser.parse_args()

    dataset = build_dataset(args)

    max_logical_samples = len(dataset)
    num_samples = min(max_logical_samples, args.max_samples)

    if num_samples == 0:
        raise RuntimeError(f"No samples found in dataset at {args.data_path}")

    # NeRF cameras look along -Z (OpenGL) → negate R to flip frustum.
    # CO3D cameras look along +Z (PyTorch3D) → use R directly.
    negate_rotation = args.dataset_type == "nerf"

    print(
        f"Dataset: {args.dataset_type}  |  clips: {len(dataset)}  |  "
        f"showing: {num_samples}"
    )

    for sample_id in range(num_samples):
        visualize_sample_images(dataset, sample_id=sample_id)
        visualize_sample_poses(
            dataset, sample_id=sample_id, negate_rotation=negate_rotation,
        )


if __name__ == "__main__":
    main()
