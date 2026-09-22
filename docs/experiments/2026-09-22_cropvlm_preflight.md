# CropVLM inference preflight, 2026-09-22

Status: waiting for the trained cropping checkpoint. No CropVLM model inference,
training, crop generation or answer generation has been run. No W&B experiment
was started because no model was executed.

The user requested the same ten single-frame questions with view/video context
as the preceding DeepEyes comparison, running on iCardio and producing a local
Markdown report with prompts, images, crops and raw outputs.

## Project identity and checkpoint search

The intended project is [CropVLM: Learning to Zoom for Fine-Grained
Vision-Language Perception](https://arxiv.org/abs/2511.19820), with
[official source](https://github.com/miguelscarv/cropvlm) pinned at
`9c9cb91379acd8e7a592553b7673fcc47f9908dc`. The clone is retained in ignored
`build/cropvlm_20260922/upstream/` locally and on iCardio. No existing submodule
pin was modified or upstream source edited.

The README requires `<CROPVLM_MODEL_PATH>` supplied by the operator. Its
example says `generate_bbox.py`; the actual file is `generate_bboxes.py`.
The full seven-commit repository was inspected, including training launchers
that refer to locally trained models. No bundled trained weights were found.
GitHub's releases API returned an empty list and the only branch is `main`.

Public Hugging Face name search returned only `boudiafA/CropVLM`, a different
[agricultural CLIP classifier](https://huggingface.co/boudiafA/CropVLM).
The author account `miguelcarv` lists eight public models, none identified as
CropVLM cropping weights. Searches for alternative name spelling found no
identified matching checkpoint. Project checkpoint/model directories and
standard Hugging Face cache names on the Mac, AMD and iCardio contained no
matching CropVLM/SmolVLM checkpoint names. These are bounded search results,
not a claim that private or differently named weights do not exist.

No generic SmolVLM base, agricultural model, DeepEyes checkpoint or fabricated
bounding box was substituted. Re-training CropVLM would be a separate task,
not equivalent to inference with the authors' checkpoint.

## Prepared inputs

The existing iCardio sample is reused unchanged:
`build/deepeyes_icardio_20260922/inputs/sample.jsonl`. Ten distinct held-out
studies, two examples per question type, selection seed 1, one random frame
from one selected view per study. Training pool: 5,061 studies; held-out pool:
1,215 studies / 31,209 rows. Membership and zero training-study overlap were
rechecked. All ten PNG hashes match the previous DeepEyes context run and
all ten images decode. One recorded view label is the literal `none`.

Prepared crop and answer prompts include the same video-frame description
and recorded view label. The upstream question capitalization is applied to
the question only; view-label case is preserved. This is an explicit context
adaptation. The prepared manifests exclude reference answers. The source
sample retains references for eventual reporting, not model input.

## Inference contract and remaining runtime work

The released crop script predicts one percentage-coordinate bounding box.
The answer script runs a separate answering model with original frame plus
crop, and also with the original alone. There is no DeepEyes-style repeated
tool conversation. The released defaults are a SmolVLM-256M processor and
answering model, longest edge 512, BF16, FlashAttention 2, and 500 new tokens
per generation. Actual trained-checkpoint identity and resolution remain
unresolved. The trained cropping model is required even though the crop
script defaults to a generic base model if its model path is omitted.

The upstream final-answer script falls back to a fixed 99-by-99-pixel box on
coordinate parsing failure. Such a fallback would need to be recorded as a
failure, not reported as a valid learned crop. No parsing/cropping adapter has
been executed or scientifically validated yet.

At preflight, iCardio GPU 1 had about 42 GiB free but was shared with another
process; no existing process was stopped. The previously isolated environment
imports `AutoModelForVision2Seq` and `AutoProcessor`, with PyTorch 2.5.1+cu121
and Transformers 4.57.3. Upstream pins PyTorch 2.6.0 and Transformers 4.57.1.
FlashAttention 2 is absent. Any SDPA substitution must be documented explicitly;
complete runtime parity and GPU model execution are not claimed. No dependencies
were installed or shared environments modified during this preflight.

## Artifacts and next requirement

[Local setup report with all ten frames and prepared prompts](../../build/cropvlm_20260922/setup.md).
The adjacent `preflight.json`, `prepared_inputs.json`, and public availability
responses preserve the actual checks. All local report links and cross-run
input hashes were validated. No model metrics exist to report.

Needed next: the trained CropVLM cropping-checkpoint link/path, plus matching
base-model/processor and trained input resolution when not included in its
metadata. Then complete the minimal inference adapter, model-load validation,
real crop/answer execution, and the final comparison report. No inference job
is currently running for this task.
