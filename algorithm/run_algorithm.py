import numpy as np
import torch

from model import UNetV2, SequenceInference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model ONCE
# =========================

model = UNetV2(
    in_channels=4,   # 和你训练时保持一致
    out_channels=1,
    base_ch=64
)

ckpt = torch.load(
    "algorithm/weights/best_model.pth",
    map_location=DEVICE
)

# 兼容你保存的是整包或 state_dict
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.to(DEVICE)
model.eval()

# 序列推理器
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
    frames: (T, H, W)      float32
    target: (H, W)         binary mask of first frame
    return: (T, H, W)      binary mask sequence
    """

    # --- 输入检查（安全，但不影响速度）---
    assert frames.ndim == 3, "frames must be (T, H, W)"
    assert target.ndim == 2, "target must be (H, W)"

    T, H, W = frames.shape

    # 归一化（与你训练一致即可）
    frames = frames.astype(np.float32)
    frames = (frames - frames.mean()) / (frames.std() + 1e-6)

    # 初始 mask -> (1,1,H,W)
    init_mask = target[None, None].astype(np.float32)

    # === 序列推理 ===
    preds = sequence_infer.infer(frames[:, None], init_mask)

    # preds: (T, H, W) bool
    return preds.astype(np.uint8)
