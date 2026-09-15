"""Pure set-difference helper for checking whether Darya's h5 caches
(report_generation/data/clip_tokens_{train,test}.h5, {train,test}_detections*.h5,
keyed by dicom_uuid) actually cover this project's own train/eval dicom pool
before any training-facing code trusts them (spec: docs/superpowers/specs/
2026-09-15-echoprime-tool-grpo-design.md, section 1's coverage risk note).
"""


def missing_dicom_uuids(needed: set, available: set) -> set:
    return needed - available
