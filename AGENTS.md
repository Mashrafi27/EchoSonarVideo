# EchoSonarVideo repository conventions

Read `CLAUDE.md` for project invariants and known training pitfalls, `SPEC.md` for research facts, and `PLAN.md` for current work. Preserve those instructions when organizing files. This file governs repository organization.

## Placement rules

Keep the existing five top-level Python packages. Do not introduce a `src/` migration or merge the two model tracks as incidental cleanup.

| Location | Contents |
| --- | --- |
| `packages/tool_env/` | Tool environment, observations, frame/view selection, zoom, budgets and parsing. |
| `packages/data_core/` | Data preparation, reward logic and SFT serialization belonging to this pipeline. Use its existing `data/`, `reward/` and `sft/` subpackages. Evaluation metrics live in `packages/eval/`, not here. |
| `packages/verl_bridge/` | verl integration, tool/session adapters and native-vision training/rollout wiring (the runtime bridge, not evaluation). |
| `packages/verl_bridge/configs/` | Existing verl YAML configurations for both tracks. Keep this canonical location until an explicit migration updates all consumers. |
| `packages/echoprime_track/` | Frozen EchoPrime encoder, projector/text model, datasets, training and rollout for the EchoPrime track. Checkpoint evaluation lives in `packages/eval/echoprime/`, not here. |
| `packages/eval/` | Unified evaluation code for BOTH tracks: shared metric primitives, NLG/BERTScore/GREEN, EchoSonar-R comparison tables (Tables 1-3), and the run/score CLI drivers. `packages/eval/echoprime/` holds the EchoPrime track's checkpoint-eval scripts specifically. Do not re-split this back into per-track `eval/` subpackages. |
| `<package>/tests/` | Tests for that package, named `test_<behavior>.py`. |
| `scripts/` | Launchers and operator commands. Put reusable logic in the package that owns it. |
| `docs/` | Human documentation, with the categories below added only when needed. |
| `external/` | Pinned submodules and their existing patch files. |
| `build/`, `outputs/`, `logs/`, `checkpoints/` | Generated local artifacts; keep them out of Git. Preserve existing runtime paths. |

The root holds project entry documents, packaging/test/dependency configuration and tool configuration. Add a root `README.md` as a concise navigation and setup guide. New papers, meeting notes, experimental scripts and run results do not belong at the root.

## Naming and grouping

- Use descriptive `snake_case` for Python files, new directories, shell scripts and experiment identifiers. Preserve conventional names such as `README.md`, `AGENTS.md` and `CLAUDE.md`.
- Name commands by action and subject, following `build_*`, `run_*`, `check_*`, `score_*` and `eval_*` where applicable.
- Put a file with the component that owns its behavior. Avoid catch-all `misc/`, `stuff/` and broad `utils.py` collections.
- Create a subpackage when several related modules have a clear responsibility. Do not create empty category directories or split cohesive modules merely to satisfy a line count.
- Do not create parallel copies named `final`, `new`, `old`, `v2` or `backup`. Use Git history for replacements; name real architectural variants by their meaning.
- Matching basenames in different packages are valid. Identify modules, model directories and checkpoints by full paths; never merge or delete them based on filename alone.

## Documentation and research records

- Keep `SPEC.md` authoritative for research facts and `PLAN.md` for current next steps. Keep unresolved issues in `docs/OPEN_ISSUES.md`.
- Keep durable operational rules in `CLAUDE.md`. Put detailed future incident narratives in `docs/troubleshooting/` and link to them; retain existing knowledge during any later extraction.
- Place meeting notes and slides in `docs/meetings/`, using `YYYY-MM-DD_<topic>` filenames.
- Place retained reference papers and their citation index in `docs/references/`.
- Place curated experiment summaries in `docs/experiments/YYYY-MM-DD_<experiment>.md`. Record the commit, model/checkpoint path, config, exact command, dataset partitions and counts, seed, metrics and W&B link when available. Do not invent missing metadata.
- Keep raw logs, caches, predictions and checkpoints in ignored artifact locations or existing external storage. Do not move them into documentation to make them tracked.
- Preserve `packages/tool_env/INTEGRATION.md` and existing configuration paths unless a deliberate move updates every reference.
- Documentation links must resolve. Mark unverified historical claims as such instead of presenting them as current results.

## Configuration and dependencies

- Keep Python configuration classes beside their owning code. Keep existing DeepSpeed JSON files in `packages/echoprime_track/` until their callers are inspected and migrated together.
- Prefer explicit CLI/config/environment inputs for machine-specific paths. Do not change operational defaults during a formatting or file-placement change.
- Preserve the `echoprime_track` installed-package metadata and `vllm.general_plugins` entry point in `pyproject.toml`. Other packages currently use the repository on `PYTHONPATH`.
- If adding nested `echoprime_track` packages, update setuptools discovery explicitly: the current package list contains only `echoprime_track`.
- Preserve submodule pins and documented patches. Do not reorganize upstream trees or remove patches based on apparent duplication.

## Cleanup procedure

1. Inspect Git status and existing instructions. Inventory exact paths and their consumers before moving anything.
2. Prepare a source-to-destination mapping. Group moves by purpose and keep scientific/model behavior unchanged.
3. For tracked generated files, confirm they are generated and preserve needed provenance first. Remove them from the Git index while retaining local copies. Adding an ignore rule alone does not untrack files. Never broadly delete working environments or experiment results.
4. Update Python imports, module invocations, shell paths, Hydra references, dynamic class/plugin strings, packaging metadata and documentation links affected by each move.
5. Run the relevant existing tests. For cross-package changes, use `python -m pytest`; current discovery covers `tool_env`, `data_core`, `verl_bridge` and `eval` (`pytest.ini` testpaths). If adding `packages/echoprime_track/tests`, add it to `pytest.ini`. Do not claim EchoPrime coverage from the existing suite.
6. Validate changed shell syntax and static references. Run model/plugin checks only in a suitable environment; report unavailable dependencies or GPU validation plainly. Do not launch training to validate documentation-only changes.
7. Review the diff for accidental data loss, changed submodule pointers, generated files and changes to experiment defaults. Report moved paths and actual validation results.

Organization work must preserve train/test partitions, reward and metric definitions, prompt/tool contracts, model registration and checkpoint compatibility. Resolve any necessary behavioral change as a separate task.
