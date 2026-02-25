import os
import copy
import time
import yaml
import torch
import torch.nn.functional as F

from app.vjepa_pose.pose_dataset import init_data
from app.vjepa_pose.transforms import make_transforms
from app.vjepa_pose.utils import load_pretrained
import src.models.vision_transformer as video_vit
from src.models.pose_predictor import vit_pose_predictor

# ------------------------------------------------------------
# Minimal single-GPU V-JEPA training script
# ------------------------------------------------------------

def main(cfg_path):

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------
    with open(cfg_path, "r") as f:
        args = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------------------
    # META
    # ------------------------------------------------------------
    cfg_meta = args["meta"]
    dtype_str = cfg_meta.get("dtype", "float32")

    if dtype_str == "float16":
        dtype = torch.float16
        use_amp = True
    else:
        dtype = torch.float32
        use_amp = False

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------
    cfg_model = args["model"]

    # Encoder (vision transformer)
    encoder = video_vit.__dict__[cfg_model["model_name"]](
        img_size=args["data"]["crop_size"],
        patch_size=args["data"]["patch_size"],
        num_frames=max(args["data"]["dataset_fpcs"]),
        tubelet_size=args["data"]["tubelet_size"],
        uniform_power=cfg_model.get("uniform_power", False),
        use_sdpa=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=cfg_model.get("use_rope", False),
    )

    # Pose-conditioned predictor
    predictor = vit_pose_predictor(
        img_size=args["data"]["crop_size"],
        patch_size=args["data"]["patch_size"],
        num_frames=max(args["data"]["dataset_fpcs"]),
        tubelet_size=args["data"]["tubelet_size"],
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=cfg_model["pred_embed_dim"],
        action_embed_dim=5,  # compact pose: (x, y, z, theta, phi)
        depth=cfg_model["pred_depth"],
        is_frame_causal=True,
        num_heads=cfg_model["pred_num_heads"],
        uniform_power=cfg_model.get("uniform_power", False),
        use_rope=cfg_model.get("use_rope", False),
        use_sdpa=False,
        use_silu=False,
        wide_silu=True,
        use_extrinsics=False,
        use_activation_checkpointing=False,
    )

    target_encoder = copy.deepcopy(encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    # -- load pretrained weights (encoder + predictor)
    p_file = cfg_meta.get("pretrain_checkpoint", None)
    context_encoder_key = cfg_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfg_meta.get("target_encoder_key", "target_encoder")
    load_predictor = cfg_model.get("load_predictor", False)
    load_encoder = cfg_model.get("load_encoder", True)
    encoder, predictor, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        target_encoder=target_encoder,
        load_predictor=load_predictor,
        load_encoder=load_encoder,
    )

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        crop_size=args["data"]["crop_size"],
    )

    loader, _ = init_data(
        data_path=args["data"]["datasets"][0],
        batch_size=args["data"]["batch_size"],
        frames_per_clip=max(args["data"]["dataset_fpcs"]),
        tubelet_size=1,
        fps=args["data"]["fps"],
        transform=transform,
        collator=torch.utils.data.default_collate,
        num_workers=0,              # IMPORTANT for stability
        world_size=1,
        pin_mem=False,
        persistent_workers=False,
        rank=0,
    )

    loader = iter(loader)

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args["optimization"]["lr"],
        weight_decay=args["optimization"]["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------
    print("Starting training...")

    tokens_per_frame = (
        args["data"]["crop_size"] // args["data"]["patch_size"]
    ) ** 2

    for step in range(100):  # small test loop

        try:
            sample = next(loader)
        except StopIteration:
            loader = iter(loader)
            sample = next(loader)

        clips = sample["buffer"].to(device)                        # [B, C, T, H, W]
        poses5 = sample["pose5"].to(device, dtype=torch.float32)    # [B, T, 5]

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):

            # ---- Target forward (no grad)
            with torch.no_grad():
                c = clips.permute(0, 2, 1, 3, 4)
                c = c.flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                h = target_encoder(c)

            # ---- Build action/state sequences from compact poses
            B, T, _ = poses5.shape
            if T <= 1:
                raise RuntimeError("Need at least 2 frames per clip for pose prediction.")

            # actions: pose at time t (last frame), broadcast in predictor
            # states:  all previous poses (frames 0 … T-2)
            actions = poses5[:, -1:, :]    # [B, 1, 5]
            states = poses5[:, :-1, :]     # [B, T-1, 5]

            # ---- Predictor forward
            z = predictor(
                h[:, :-tokens_per_frame],
                actions,
                states,
            )

            loss = torch.mean(torch.abs(z))

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    print("Finished training test.")


if __name__ == "__main__":
    main("configs/train/vitl16/pose-256px-8f.yaml")
