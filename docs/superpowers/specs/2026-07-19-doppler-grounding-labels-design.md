# Visually-Grounded Reasoning Labels for Echo VQA — Colour-Doppler Increment

**Date:** 2026-07-19
**Scope:** First build increment. Flow-predicate questions only, colour-Doppler box + phase window.
**Out of scope (deferred):** chamber segmentation labels, keypoints, ED/ES indices, reward formulation, RL training.

## Purpose

Generate spatial and temporal grounding labels for the echo VQA dataset automatically, so that a
downstream RL policy can be rewarded for attending to the right region of the right clip rather
than for answer text alone.

This increment builds the full pipeline end-to-end for the one label source that has no external
dependencies: colour-Doppler pixel extraction. The chamber-segmentation source slots in later
behind the same interface.

## Inputs

| Path | Contents |
|---|---|
| `EchoSonarVideo/Archive 2 (1)/train_vqa_with_thinking.jsonl` | 128,215 QA pairs, 5,061 studies, 104 distinct questions, `thinking` field |
| `EchoSonarVideo/Archive 2 (1)/test_vqa.jsonl` | 31,209 QA pairs, 1,215 studies, no `thinking` field |
| `studies_json/combined/combined_updated.json` | `studies` (79,584) + `dicoms` (4.1M: `dicom_uuid`, `study_uuid`, `view`, `n_frames`, `physical_delta_x/y`, `type`) |
| `studies_json/visibility/visibility_matrix(final_matrix).csv` | 68 views × 12 structures, weights in {0, 0.1, 0.3, 0.5, 0.8, 1} |
| `preprocessed_data_doppler/<study>/<dicom>_<view>/N.png` | 336×336 RGB colour-flow frames |

## Established facts

These were measured, not assumed. They constrain the design.

- **The question set is closed.** 104 distinct question strings: 48 `structure_description`
  (12 structures × 4 paraphrases), 44 `abnormality_classification` (11 concepts × 4 paraphrases),
  12 report/conclusion/list variants. Routing is a static table, not a parsing problem.
- **The thinking traces select clips reliably.** View names extracted from the trace and matched
  against the study's real DICOM views score **precision 0.985, recall 0.208** over 4,000 samples;
  per-sample precision is 1.0 on 3,629/3,999. The trace names a genuine subset of the study's clips.
- **The traces' clinical justifications are not verified.** They are post-hoc LLM text. Used for
  clip routing only, never as reasoning ground truth.
- **Colour flow is chromatically separable.** Saturation threshold + morphological open/close +
  largest connected component yields stable, anatomically plausible boxes
  (e.g. `A3C Color on AV` → median box `[161 108 215 224]`, x-jitter 1.5 px across frames).
- **Doppler view labels cover the valves.** ~1/3 of Doppler clips carry an explicit
  `<view> Color on <structure>` label. Study-level coverage: MV 71.6%, AV 76.3%, TV 51.7%, PV 31.9%.
- **Only four measurements are populated.** LVEDD 99.1%, LVESD 98.9%, LA-dim 98.3%, EF 94.3%.
  Every Doppler-derived measurement field (TR velocity, MV E/A, LVOT VTI, gradients, EPSS) is 0%.
- **Answers are heavily imbalanced.** 77% "No" overall; per concept 4% (bicuspid AV) to 66% (TR).
  The flow concepts targeted by this increment are the best balanced: MR 60–64%, TR 64–66%,
  AR 22–25%.
- **Splits are clean on studies, not on designation.** Zero study overlap between train and test.
  The test file mixes official designations: 1,020 `TEST`, 153 `TRAIN`, 42 `VAL`.

## Architecture

Four units, each independently testable, communicating through plain serialized records.

```
  QA pair
     |
     v
[1] question_router      question string -> (structure, predicate, required_modality)
     |
     v
[2] clip_selector        thinking trace + study DICOM index + visibility matrix
     |                   -> ordered list of admissible clips
     v
[3] doppler_grounder     clip frames -> spatial box + phase window (or None)
     |
     v
[4] label_writer         -> grounded label record (JSONL)
```

### [1] `question_router`

A static table over the 104 question strings. Emits:

- `structure` — one of the 12 report structures
- `predicate` — `size | flow | morphology | function`
- `required_modality` — `bmode | color`

`flow` + `color` is the subset this increment grounds. All other predicates emit a record with
`grounding: null` and a reason code, so the coverage gap is explicit in the output rather than
silent.

The table is authored once by hand and checked by an assertion that every question string in both
JSONL files resolves. Unresolved strings are a hard error, not a fallback.

### [2] `clip_selector`

1. Extract view names from the `thinking` field by matching against the 68-term view vocabulary
   (longest-match first, word-boundary anchored).
2. Intersect with the study's actual clips from the DICOM index.
3. Filter to clips where `visibility_matrix[view][structure] > 0`.
4. Filter to `dicom_type == "Color"` when `required_modality == color`.
5. Rank by visibility weight descending.

For the test set, which has no `thinking` field, step 1 is skipped and all of the study's clips
enter at step 2. This asymmetry is deliberate and must be recorded in the output record, because a
train/test difference in clip selection is a confound if left implicit.

### [3] `doppler_grounder`

Per frame:

1. `sat = max(RGB) - min(RGB)`; mask `= (sat > 50) & (max(RGB) > 60)`
2. Binary opening (3×3), then closing (7×7)
3. Largest connected component; reject if area below a floor
4. Box = component extent; centroid = component mean (this is the *pointing* label)

Per clip:

- Spatial box = per-frame median of accepted boxes
- Phase window = the contiguous frame interval where component area exceeds a fraction of its
  own maximum, giving the interval over which flow is present

Thresholds are parameters with the values above as defaults, not constants.

**Known failure mode to handle explicitly:** the on-screen colour reference bar is saturated and
will be picked up by the threshold. The largest-connected-component step removes it in the cases
tested, but it must be guarded — an ROI mask excluding the frame margin, validated by checking that
box centroids do not cluster at the image edge.

### [4] `label_writer`

One JSONL record per input QA pair, preserving the original fields and adding:

```json
{
  "study_uuid": "...", "question_type": "...", "messages": [...],
  "routing":  {"structure": "mitral_valve", "predicate": "flow", "required_modality": "color"},
  "clips":    [{"dicom_uuid": "...", "view": "A4C Color on MV", "visibility": 1.0}],
  "grounding": {
    "source": "color_doppler",
    "box": [155, 150, 208, 207],
    "point": [181, 178],
    "phase_window": [7, 19],
    "n_frames": 25
  },
  "flags": ["jet_not_anatomy"]
}
```

`grounding: null` plus a reason code when no admissible source exists.

## Semantics to state, not discover later

The colour-Doppler box localizes the **jet, not the valve anatomy**. For mitral regurgitation the
box sits in the left atrium, because that is where regurgitant flow travels. This is diagnostically
correct — it is where a reader looks for the pathology — but it is not a box on the mitral valve,
and any downstream reward or paper text must describe it as such. Hence the persistent
`jet_not_anatomy` flag on every colour-Doppler record.

## Error handling

- Unresolvable question string → hard error (the closed-set assumption is load-bearing)
- Study missing from DICOM index → record with `grounding: null`, reason `study_not_indexed`
- No admissible clip → `grounding: null`, reason `no_admissible_clip`
- No colour component found in any frame → `grounding: null`, reason `no_flow_detected`
- Corrupt/missing frame → skip the frame, proceed if any frames remain

No silent drops. Every input QA pair produces exactly one output record. Counts of each reason code
are reported at the end of a run, so coverage is visible rather than inferred.

## Testing

- `question_router`: every question string in both JSONL files resolves; paraphrase groups map to
  identical triples (the 4 phrasings of each concept must agree)
- `clip_selector`: on a sample with known views, selected clips are a subset of the study's real
  clips; reproduces the measured 0.985 precision
- `doppler_grounder`: on hand-checked clips, box centroids fall within the central image region and
  not at the margin (colour-bar guard); boxes are stable across adjacent frames
- End-to-end: output record count equals input line count; reason-code histogram sums correctly

## Success criteria

1. Every QA pair in both files produces exactly one output record.
2. Flow-predicate questions with an admissible colour clip receive a box, point, and phase window.
3. Coverage is reported per concept and matches the study-level estimates (MV ~72%, AV ~76%,
   TV ~52%, PV ~32%) within a reasonable margin.
4. No record claims grounding it does not have.

## Deferred

- Chamber segmentation labels (CAMUS/EchoNet — needs sourcing; no model on this machine)
- Measurement-validated keypoints for `size` predicates
- ED/ES indices for `function` predicates
- Recovering the ~2/3 of Doppler clips whose view is `none`/`Unknown` via the view classifier
  (`dicom_view_mapping_202510241152.csv`)
- Reward formulation, including per-concept balancing to defeat the 77%-"No" degenerate policy
- Re-cutting train/test against `study_designation`
