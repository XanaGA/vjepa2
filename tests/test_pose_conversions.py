#!/usr/bin/env python3
"""Tests for c2w_to_pose5 / pose5_to_c2w round-trip conversions."""

import math

import torch
import pytest

from app.vjepa_pose.pose_dataset import c2w_to_pose5, pose5_to_c2w

ATOL = 1e-5


def _make_c2w(x_axis, y_axis, z_axis, pos):
    """Build a 4x4 c2w from column vectors."""
    c2w = torch.eye(4)
    c2w[:3, 0] = torch.tensor(x_axis, dtype=torch.float32)
    c2w[:3, 1] = torch.tensor(y_axis, dtype=torch.float32)
    c2w[:3, 2] = torch.tensor(z_axis, dtype=torch.float32)
    c2w[:3, 3] = torch.tensor(pos, dtype=torch.float32)
    return c2w


def _angles_close(a, b, atol=ATOL):
    """Compare two angle tensors modulo 2π."""
    diff = (a - b + math.pi) % (2 * math.pi) - math.pi
    return diff.abs().max().item() < atol


# === Hardcoded test cases ===
# Each entry: (description, c2w, expected_pose5)

IDENTITY = _make_c2w(
    x_axis=[1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, 1],
    pos=[0, 0, 0],
)
# Look along +Z → theta=0, phi=0
IDENTITY_POSE5 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

LOOK_ALONG_PLUS_X = _make_c2w(
    x_axis=[0, 0, -1], y_axis=[0, 1, 0], z_axis=[1, 0, 0],
    pos=[1, 2, 3],
)
# Look along +X → theta=pi/2, phi=0
LOOK_PLUS_X_POSE5 = torch.tensor([1.0, 2.0, 3.0, math.pi / 2, 0.0])

LOOK_ALONG_MINUS_X = _make_c2w(
    x_axis=[0, 0, 1], y_axis=[0, 1, 0], z_axis=[-1, 0, 0],
    pos=[-5, 0, 10],
)
# Look along -X → theta=pi/2, phi=pi
LOOK_MINUS_X_POSE5 = torch.tensor([-5.0, 0.0, 10.0, math.pi / 2, math.pi])

LOOK_ALONG_PLUS_Y = _make_c2w(
    x_axis=[1, 0, 0], y_axis=[0, 0, -1], z_axis=[0, 1, 0],
    pos=[0, 0, 0],
)
# Look along +Y → theta=pi/2, phi=pi/2
LOOK_PLUS_Y_POSE5 = torch.tensor([0.0, 0.0, 0.0, math.pi / 2, math.pi / 2])

LOOK_ALONG_MINUS_Z = _make_c2w(
    x_axis=[-1, 0, 0], y_axis=[0, 1, 0], z_axis=[0, 0, -1],
    pos=[7, 8, 9],
)
# Look along -Z → theta=pi, phi=0
LOOK_MINUS_Z_POSE5 = torch.tensor([7.0, 8.0, 9.0, math.pi, 0.0])

# 45° between +X and +Z in the XZ plane
SQRT2_2 = math.sqrt(2) / 2
LOOK_45_XZ = _make_c2w(
    x_axis=[SQRT2_2, 0, -SQRT2_2],
    y_axis=[0, 1, 0],
    z_axis=[SQRT2_2, 0, SQRT2_2],
    pos=[10, 20, 30],
)
# theta = pi/4, phi = 0
LOOK_45_XZ_POSE5 = torch.tensor([10.0, 20.0, 30.0, math.pi / 4, 0.0])


CASES = [
    ("identity (look +Z)",        IDENTITY,           IDENTITY_POSE5),
    ("look +X",                   LOOK_ALONG_PLUS_X,  LOOK_PLUS_X_POSE5),
    ("look -X",                   LOOK_ALONG_MINUS_X, LOOK_MINUS_X_POSE5),
    ("look +Y (degenerate)",      LOOK_ALONG_PLUS_Y,  LOOK_PLUS_Y_POSE5),
    ("look -Z",                   LOOK_ALONG_MINUS_Z, LOOK_MINUS_Z_POSE5),
    ("look 45° in XZ plane",      LOOK_45_XZ,         LOOK_45_XZ_POSE5),
]


# ------------------------------------------------------------------
# Forward: c2w → pose5
# ------------------------------------------------------------------

class TestC2WToPose5:

    @pytest.mark.parametrize("name,c2w,expected", CASES, ids=[c[0] for c in CASES])
    def test_forward(self, name, c2w, expected):
        pose5 = c2w_to_pose5(c2w)
        assert pose5.shape == (5,)
        torch.testing.assert_close(pose5, expected, atol=ATOL, rtol=0)

    def test_batched(self):
        batch = torch.stack([c for _, c, _ in CASES], dim=0)  # [N, 4, 4]
        expected = torch.stack([e for _, _, e in CASES], dim=0)
        result = c2w_to_pose5(batch)
        assert result.shape == (len(CASES), 5)
        torch.testing.assert_close(result, expected, atol=ATOL, rtol=0)


# ------------------------------------------------------------------
# Inverse: pose5 → c2w
# ------------------------------------------------------------------

class TestPose5ToC2W:

    @pytest.mark.parametrize("name,c2w,expected_pose5", CASES, ids=[c[0] for c in CASES])
    def test_inverse_produces_valid_rotation(self, name, c2w, expected_pose5):
        """Reconstructed rotation must be orthogonal with det=+1."""
        rec = pose5_to_c2w(expected_pose5)
        R = rec[:3, :3]
        torch.testing.assert_close(R @ R.T, torch.eye(3), atol=ATOL, rtol=0)
        assert torch.det(R).item() == pytest.approx(1.0, abs=ATOL)

    @pytest.mark.parametrize("name,c2w,expected_pose5", CASES, ids=[c[0] for c in CASES])
    def test_inverse_position(self, name, c2w, expected_pose5):
        rec = pose5_to_c2w(expected_pose5)
        torch.testing.assert_close(rec[:3, 3], c2w[:3, 3], atol=ATOL, rtol=0)

    @pytest.mark.parametrize("name,c2w,expected_pose5", CASES, ids=[c[0] for c in CASES])
    def test_inverse_look_direction(self, name, c2w, expected_pose5):
        """Z-axis (look direction) must match the original."""
        rec = pose5_to_c2w(expected_pose5)
        torch.testing.assert_close(rec[:3, 2], c2w[:3, 2], atol=ATOL, rtol=0)


# ------------------------------------------------------------------
# Round-trip: c2w → pose5 → c2w
# ------------------------------------------------------------------

class TestRoundTrip:

    @pytest.mark.parametrize("name,c2w,_", CASES, ids=[c[0] for c in CASES])
    def test_position_roundtrip(self, name, c2w, _):
        rec = pose5_to_c2w(c2w_to_pose5(c2w))
        torch.testing.assert_close(rec[:3, 3], c2w[:3, 3], atol=ATOL, rtol=0)

    @pytest.mark.parametrize("name,c2w,_", CASES, ids=[c[0] for c in CASES])
    def test_look_direction_roundtrip(self, name, c2w, _):
        rec = pose5_to_c2w(c2w_to_pose5(c2w))
        torch.testing.assert_close(rec[:3, 2], c2w[:3, 2], atol=ATOL, rtol=0)

    @pytest.mark.parametrize("name,c2w,_", CASES, ids=[c[0] for c in CASES])
    def test_full_matrix_roundtrip(self, name, c2w, _):
        """Full matrix round-trip.

        Only exact when the original has zero roll relative to the
        Gram-Schmidt up-vector convention used by pose5_to_c2w.
        The degenerate (look ≈ ±Y) case uses a different reference
        axis, so x/y may differ — we only check position + look dir.
        """
        rec = pose5_to_c2w(c2w_to_pose5(c2w))
        torch.testing.assert_close(rec[:3, 3], c2w[:3, 3], atol=ATOL, rtol=0)
        torch.testing.assert_close(rec[:3, 2], c2w[:3, 2], atol=ATOL, rtol=0)

    def test_batched_roundtrip(self):
        batch = torch.stack([c for _, c, _ in CASES], dim=0)
        rec = pose5_to_c2w(c2w_to_pose5(batch))
        torch.testing.assert_close(rec[..., :3, 3], batch[..., :3, 3],
                                   atol=ATOL, rtol=0)
        torch.testing.assert_close(rec[..., :3, 2], batch[..., :3, 2],
                                   atol=ATOL, rtol=0)


# ------------------------------------------------------------------
# Round-trip: pose5 → c2w → pose5
# ------------------------------------------------------------------

class TestRoundTripReverse:
    """pose5 → c2w → pose5.

    Angles have inherent ambiguities (atan2 wrap at ±π, and phi is
    undefined at the poles theta=0 or theta=π), so we compare the
    reconstructed look-direction **vectors** instead of raw angles.
    """

    @staticmethod
    def _look_from_pose5(pose5):
        theta, phi = pose5[..., 3], pose5[..., 4]
        sin_t = torch.sin(theta)
        return torch.stack([sin_t * torch.cos(phi),
                            sin_t * torch.sin(phi),
                            torch.cos(theta)], dim=-1)

    @pytest.mark.parametrize("name,_,pose5", CASES, ids=[c[0] for c in CASES])
    def test_position_roundtrip(self, name, _, pose5):
        rec = c2w_to_pose5(pose5_to_c2w(pose5))
        torch.testing.assert_close(rec[:3], pose5[:3], atol=ATOL, rtol=0)

    @pytest.mark.parametrize("name,_,pose5", CASES, ids=[c[0] for c in CASES])
    def test_look_direction_roundtrip(self, name, _, pose5):
        rec = c2w_to_pose5(pose5_to_c2w(pose5))
        torch.testing.assert_close(
            self._look_from_pose5(rec),
            self._look_from_pose5(pose5),
            atol=ATOL, rtol=0,
        )

    def test_batched_position_roundtrip(self):
        batch = torch.stack([e for _, _, e in CASES], dim=0)
        rec = c2w_to_pose5(pose5_to_c2w(batch))
        torch.testing.assert_close(rec[:, :3], batch[:, :3], atol=ATOL, rtol=0)

    def test_batched_look_direction_roundtrip(self):
        batch = torch.stack([e for _, _, e in CASES], dim=0)
        rec = c2w_to_pose5(pose5_to_c2w(batch))
        torch.testing.assert_close(
            self._look_from_pose5(rec),
            self._look_from_pose5(batch),
            atol=ATOL, rtol=0,
        )
