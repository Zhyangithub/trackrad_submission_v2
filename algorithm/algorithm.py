import numpy as np
import torch

from model import UNetV2, SequenceInference

# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load model ONCE (IMPORTANT)
# =========================
MODEL_PATH = "/opt/ml/model/best_model.pth"

model = UNetV2(
    in_channels=4,
    out_channels=1,
    base_ch=64
)

state = torch.load(MODEL_PATH, map_location=DEVICE)

# 兼容 state_dict / checkpoint
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

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
    frames: (T, H, W) float32
    target: (H, W) binary
    return: (T, H, W) binary
    """

    # ---- safety checks ----
    assert frames.ndim == 3
    assert target.ndim == 2

    frames = frames.astype(np.float32)
    target = target.astype(np.uint8)

    # ---- inference ----
    preds = sequence_infer.predict_sequence(frames, target)

    return preds.astype(np.uint8)
