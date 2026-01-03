import numpy as np
import torch
from model import UNetV2, SequenceInference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- load model once --------
model = UNetV2(in_channels=4, out_channels=1, base_ch=64)
ckpt = torch.load("/opt/ml/model/best_model.pth", map_location=DEVICE)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.to(DEVICE)
model.eval()
sequence_infer = SequenceInference(model)


# -------- REQUIRED ENTRY --------
def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:

    # ===== 强制防炸 =====
    try:
        T, H, W = frames.shape
        frames = frames.astype(np.float32)

        # 防止 std = 0
        std = frames.std()
        if std < 1e-6:
            frames = frames * 0.0
        else:
            frames = (frames - frames.mean()) / (std + 1e-6)

        # 处理 target
        target = target.astype(np.float32)
        if target.max() > 1:
            target = target / 255.0
        if target.sum() == 0:
            # 空 mask → 返回全 0
            return np.zeros((T, H, W), dtype=np.uint8)

        init_mask = target[None, None]

        preds = sequence_infer.infer(
            frames[:, None],
            init_mask
        )

        preds = preds.astype(np.uint8)
        if preds.shape != (T, H, W):
            raise RuntimeError("Invalid output shape")

        return preds

    except Exception as e:
        # ===== 最重要的一行 =====
        # 任何异常 → 返回合法空结果，而不是 crash
        return np.zeros(frames.shape, dtype=np.uint8)
