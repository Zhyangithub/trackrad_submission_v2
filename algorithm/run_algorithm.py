import numpy as np
import torch

from model import UNetV2, SequenceInference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model ONCE
# =========================

model = UNetV2(
    in_channels=4,
    out_channels=1,
    base_ch=64
)

ckpt = torch.load(
    "/opt/ml/model/best_model.pth",
    map_location=DEVICE
)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.to(DEVICE)
model.eval()

sequence_infer = SequenceInference(model)

# =========================
# TrackRAD REQUIRED ENTRY
# =========================

def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """
    frames: (T, H, W) or (T, 1, H, W)
    target: (H, W) or (1, H, W)
    return: (T, H, W)
    """

    # -------- 1. 统一 frames 维度 --------
    if frames.ndim == 4:
        frames = frames[:, 0]      # (T, H, W)
    elif frames.ndim != 3:
        raise ValueError(f"Unexpected frames shape: {frames.shape}")

    # -------- 2. 统一 target 维度 --------
    if target.ndim == 3:
        target = target[0]
    elif target.ndim != 2:
        raise ValueError(f"Unexpected target shape: {target.shape}")

    T, H, W = frames.shape

    # -------- 3. 归一化 --------
    frames = frames.astype(np.float32)
    frames = (frames - frames.mean()) / (frames.std() + 1e-6)

    # -------- 4. 准备模型输入 --------
    frames_torch = torch.from_numpy(frames).to(DEVICE)
    frames_torch = frames_torch[:, None]        # (T, 1, H, W)

    init_mask = torch.from_numpy(
        target[None, None].astype(np.float32)
    ).to(DEVICE)

    # -------- 5. 推理 --------
    with torch.no_grad():
        preds = sequence_infer.infer(frames_torch, init_mask)

    return preds.astype(np.uint8)
