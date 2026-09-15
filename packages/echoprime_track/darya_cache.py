"""Shared utilities for reading Darya's precomputed h5 caches directly, lazily, per dicom_uuid --
the SAME access pattern her own training code uses (report_generation/sft_thinking/dataset.py::
EchoVQAThinkingDataset), proven at real training scale (her 0.795/0.799 GREEN numbers came from
exactly this). No intermediate per-study cache needed -- reading a single (393,768) or (n,256)
slice out of an h5 file is fast even over a network filesystem, and h5py supports concurrent
reads from multiple processes fine. Used by generate_grpo_parquet.py (Task 7, for prompt-text
placeholder counts) and echoprime_tool_agent_loop.py (Task 8, for the actual tensor data at
rollout time) -- one source of truth for both, instead of duplicating the h5-reading logic.
"""
import numpy as np

# Same 7 RT-DETR structure classes report_generation/sft_thinking/dataset.py::RT_DETR_CLASSES
# uses -- kept in sync manually (small, stable, not worth a cross-repo import).
RT_DETR_CLASSES = {
    0: "Left Ventricle", 1: "Left Atrium", 2: "Right Atrium", 3: "Right Ventricle",
    4: "Mitral Valve", 5: "Tricuspid Valve", 6: "LVOT Area",
}
NUM_FRAMES = 16


def load_clip_tokens(clip_h5, dicom_uuid: str):
    """(393, 768) numpy array, or None if this dicom isn't in the cache (spec section 1: this
    is EXPECTED to happen for a real fraction of dicoms -- callers must handle None, not treat
    it as an error)."""
    if dicom_uuid not in clip_h5:
        return None
    return clip_h5[dicom_uuid]["tokens"][:]


def detr_class_ids_present(detr_h5, dicom_uuid: str) -> list:
    if dicom_uuid not in detr_h5:
        return []
    grp = detr_h5[dicom_uuid]
    classes = set()
    for frame_idx in range(NUM_FRAMES):
        key = f"frame_{frame_idx}"
        if key not in grp:
            continue
        for cls_id in grp[key]["classes"][:]:
            classes.add(int(cls_id))
    return sorted(classes)


def load_detr_tokens(detr_h5, dicom_uuid: str) -> np.ndarray:
    """(n_classes_present, 256) mean-pooled-over-16-frames-per-class, sorted by class id --
    mirrors report_generation/sft_thinking/dataset.py::EchoVQAThinkingDataset.
    _load_detr_structure_tokens exactly (same mean-pooling, same sort order). Empty (0, 256)
    array if no detections at all for this dicom."""
    class_ids = detr_class_ids_present(detr_h5, dicom_uuid)
    if not class_ids:
        return np.zeros((0, 256), dtype=np.float32)
    grp = detr_h5[dicom_uuid]
    pooled = []
    for target_cls in class_ids:
        embeds = []
        for frame_idx in range(NUM_FRAMES):
            key = f"frame_{frame_idx}"
            if key not in grp:
                continue
            classes = grp[key]["classes"][:]
            frame_embeds = grp[key]["embeddings"][:]
            for cls_id, emb in zip(classes, frame_embeds):
                if int(cls_id) == target_cls:
                    embeds.append(emb)
        pooled.append(np.mean(embeds, axis=0))
    return np.stack(pooled).astype(np.float32)
