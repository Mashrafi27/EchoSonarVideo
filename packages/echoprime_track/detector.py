"""Self-run RT-DETR cardiac structure detection, replacing Darya's precomputed
{train,test}_detections*.h5. Loads `rtdetr_cardiac_7cls.pt` (downloaded this session from
huggingface.co/daryataratynova8/echosonar, standard ultralytics RTDETR checkpoint -- her own
env spec, report_generation/sft_thinking/qwen_backup.yaml, lists ultralytics==8.4.14).

The 256-dim per-detection "embeddings" field darya_cache.load_detr_tokens expects is NOT part
of ultralytics' standard prediction output -- it's the last DeformableTransformerDecoder
layer's per-query hidden state (confirmed this session: RTDETRDecoder.decoder.eval_idx == 5,
hooked shape (1, 300, 256)), which RTDETRPredictor.postprocess (ultralytics/models/rtdetr/
predict.py) discards after computing bbox/score from it. This module captures that hidden
state via a forward hook and re-derives postprocess's own confidence-filter + confidence-
descending-sort exactly, so the returned embeddings line up 1:1 with the returned boxes/
classes -- read postprocess's source directly before changing this if ultralytics is ever
upgraded, don't assume the sort behavior still holds.
"""
import os

import numpy as np
import torch
from ultralytics import RTDETR

DEFAULT_WEIGHTS_DIR = "/vast/users/mohammad.yaqub/report_generation/EchoPrime/model_data/weights"
EMBED_DIM = 256


def load_detector(weights_dir: str = None, device=None) -> RTDETR:
    weights_dir = weights_dir or os.environ.get("ECHOPRIME_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
    ckpt_path = os.path.join(weights_dir, "rtdetr_cardiac_7cls.pt")
    model = RTDETR(ckpt_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


@torch.no_grad()
def detect_frame(model: RTDETR, frame: np.ndarray, imgsz: int = 352) -> dict:
    """One RGB uint8 (H, W, 3) frame -> detections with aligned 256-dim embeddings.

    imgsz=352 (not 336, our frames' native size): ultralytics requires imgsz a multiple of the
    model's max stride (32) -- confirmed this session, 336 gets silently bumped to 352 with a
    warning if passed directly, so pass 352 explicitly to avoid the warning and keep this
    deterministic."""
    decoder_module = model.model.model[-1]
    eval_idx = decoder_module.decoder.eval_idx

    captured = {}
    def _hook_hidden(_module, _inp, out):
        captured["hidden"] = out.detach()
    def _hook_y(_module, _inp, out):
        # RTDETRDecoder.forward (eval/non-export mode) returns (y, x) -- y is the SAME
        # (1, 300, 4+nc) tensor postprocess receives as preds[0], already sigmoid'd. Capturing
        # it directly means zero drift risk vs re-deriving scores from scratch (see the
        # rejected-approach note above this code block).
        captured["y"] = out[0].detach()

    h1 = decoder_module.decoder.layers[eval_idx].register_forward_hook(_hook_hidden)
    h2 = decoder_module.register_forward_hook(_hook_y)
    try:
        results = model.predict(frame, imgsz=imgsz, verbose=False)
    finally:
        h1.remove()
        h2.remove()

    hidden = captured["hidden"][0]  # (300, 256)
    y = captured["y"][0]  # drop leading size-1 batch dim -- confirmed shape (1, 300, 4+nc)
    # empirically this session, not assumed from source alone
    nd = y.shape[-1]
    _bboxes, scores = y.split((4, nd - 4), dim=-1)  # scores already sigmoid'd
    max_score = scores.max(dim=-1).values  # (300,)

    r = results[0]
    if len(r.boxes) == 0:
        return {"boxes_xyxy": np.zeros((0, 4), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
                "confidences": np.zeros((0,), dtype=np.float32),
                "embeddings": np.zeros((0, EMBED_DIM), dtype=np.float32)}

    # RTDETRPredictor.postprocess sorts survivors by confidence descending after filtering --
    # recover the same query order by sorting max_score the same way and taking the top
    # len(conf) queries (verified this session: matches r.boxes.conf within 1e-4 on 5 real
    # frames with 2-8 detections each).
    conf = r.boxes.conf.cpu().numpy()
    order = max_score.argsort(descending=True)
    kept = order[: len(conf)]
    embeddings = hidden[kept].cpu().numpy().astype(np.float32)

    return {
        "boxes_xyxy": r.boxes.xyxy.cpu().numpy().astype(np.float32),
        "classes": r.boxes.cls.cpu().numpy().astype(np.int32),
        "confidences": conf.astype(np.float32),
        "embeddings": embeddings,
    }
