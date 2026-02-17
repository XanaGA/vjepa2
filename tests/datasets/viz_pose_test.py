#!/usr/bin/env python3
#
# Utility script to visualize PoseVideoDataset samples:
#  - RGB images for a subset of samples
#  - Corresponding camera poses in 3D using Open3D
#

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

from app.vjepa_pose.pose_dataset import PoseVideoDataset


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
    # frame_indices = frame_indices[:max_frames]
    # n = len(frame_indices)
    n = max_frames
    if n == 0:
        return

    cols = min(n, max_frames)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    buffer, poses = dataset[sample_id]  # buffer: [T, C, H, W], poses: [T, 4, 4]
    buffer = buffer.permute(1,0,2,3)  # buffer: [C, T, H, W]

    for ax, idx in zip(axes, range(len(buffer))):
        
        img = buffer[idx].permute(1, 2, 0)  # [H, W, C]
        img_np = img.cpu().numpy()

        # If the tensor is not uint8 (e.g., after normalization), try to bring it
        # back to a displayable range.
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

    # Turn off any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Sample {sample_id}: first {n} frames")
    plt.tight_layout()
    plt.show()


def visualize_sample_poses(dataset, sample_id):
    """
    Visualize camera poses in 3D for a single logical sample using Open3D.

    For each frame index we create a camera frustum mesh and place it using:
        camera_mesh.translate(t)
        camera_mesh.rotate(R, center=t)
    where R, t come from the 4x4 pose matrix.
    """
    geometries = []

    # Add world coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    _, poses = dataset[sample_id]  # poses: [T, 4, 4]


    for idx in range(len(poses)):
        pose = poses[idx]  # T=1

        pose_np = pose.cpu().numpy()
        R = pose_np[:3, :3]
        t = pose_np[:3, 3]

        camera_mesh = create_camera_mesh(scale=0.2, color=(0.1, 0.7, 0.9))
        camera_mesh.translate(t)  # Apply translation
        camera_mesh.rotate(-R, center=t)  # Apply -rotation (- bc reference system)

        geometries.append(camera_mesh)

    if len(geometries) > 1:
        o3d.visualization.draw_geometries(geometries)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize PoseVideoDataset samples: show images with matplotlib and "
            "camera poses in 3D with Open3D."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to NeRF synthetic scene root (e.g. ~/Downloads/nerf_synthetic/lego).",
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
        help="Number of consecutive frames to group into one logical sample.",
    )
    args = parser.parse_args()

    data_path = os.path.expanduser(args.data_path)
    if not os.path.isdir(data_path):
        raise NotADirectoryError(f"Invalid data path: {data_path}")

    # Use no transform so we see raw images as stored on disk.
    dataset = PoseVideoDataset(data_path=data_path, transform=None, frames_per_clip=args.frames_per_sample)

    # Define how many logical samples we can get by grouping consecutive frames.
    # frames_per_sample = max(1, int(args.frames_per_sample))
    max_logical_samples = len(dataset)
    num_samples = min(max_logical_samples, args.max_samples)

    if num_samples == 0:
        raise RuntimeError(f"No samples found in dataset at {data_path}")

    for sample_id in range(num_samples):
        # start_idx = sample_id * frames_per_sample
        # frame_indices = list(range(start_idx, min(start_idx + frames_per_sample, len(dataset))))

        # 1) Per-sample: visualize images (first N frames)
        visualize_sample_images(dataset, sample_id=sample_id)

        # 2) Per-sample: visualize corresponding poses in 3D with Open3D
        visualize_sample_poses(dataset, sample_id=sample_id)


if __name__ == "__main__":
    main()

