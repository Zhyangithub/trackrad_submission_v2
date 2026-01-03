import numpy as np
import torch

from model import UNetV2, SequenceInference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetV2(in_channels=4, out_channels=1, base_ch=64)

ckpt = torch.load("/opt/ml/model/best_model.pth", map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.to(DEVICE)
model.eval()

sequence_infer = SequenceInference(model)


def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:

    try:
        # ---------- frames ----------
        if frames.ndim == 4:
            frames = frames[:, 0]
        if frames.ndim != 3:
            raise ValueError(f"frames shape invalid: {frames.shape}")

        # ---------- target ----------
        if target.ndim == 3:
            target = target[0]
        if target.ndim != 2:
            raise ValueError(f"target shape invalid: {target.shape}")

        T, H, W = frames.shape

        # ---------- 空 mask 兜底 ----------
        if np.sum(target) == 0:
            return np.zeros((T, H, W), dtype=np.uint8)

        # ---------- normalize ----------
        frames = frames.astype(np.float32)
        std = frames.std()
        if std < 1e-6:
            return np.zeros((T, H, W), dtype=np.uint8)

        frames = (frames - frames.mean()) / (std + 1e-6)

        frames_t = torch.from_numpy(frames).to(DEVICE)[:, None]
        init_mask = torch.from_numpy(
            target[None, None].astype(np.float32)
        ).to(DEVICE)

        with torch.no_grad():
            preds = sequence_infer.infer(frames_t, init_mask)

        preds = np.nan_to_num(preds)
        return preds.astype(np.uint8)

    except Exception as e:
        # ❗评测期绝不能抛异常
        return np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.uint8)
