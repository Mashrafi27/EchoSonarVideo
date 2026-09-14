---
name: repo-cleanup
description: Audit and reorganize a codebase into understandable folders with consistent naming and clear ownership. Use for repository organization, file placement, and housekeeping. Update affected references and verify behavior.
---

# Repository Cleanup

Help the user maintain an organized codebase without requiring them
to understand or design its folder structure.

## Understand the project

- Read AGENTS.md, CLAUDE.md, and applicable nested instructions.
- Inspect the requested branch, working-tree changes, package
  configuration, test discovery, launchers, and documentation.
- Inventory code, configuration, tests, documentation, dependencies,
  and generated artifacts.
- Inspect actual imports and callers before deciding where files belong.
- Explain the existing folders in a short table using plain language.

## Choose a clearer structure

- Treat the current layout as something to evaluate, not preserve
  automatically.
- Recommend a simple layout based on each component's responsibility.
- Keep related files together and give each category one canonical home.
- Avoid miscellaneous dumping folders, unnecessary nesting, empty
  categories, and copies named final, new, old, or backup.
- Use descriptive snake_case names for Python files and directories.
  Preserve conventional names such as README.md and AGENTS.md.
- Identify models and checkpoints by full path. Matching filenames
  in different folders do not establish duplication.
- Show a current-to-proposed path table with a brief reason for each move.
- Make routine organization decisions yourself. Do not require the user
  to choose between technical folder architectures.

## Apply the requested cleanup

For audit-only requests, stop at findings and recommendations.
For cleanup requests, proceed with authorized reversible changes.

- Preserve unrelated edits.
- Move one coherent group at a time.
- Update imports, module commands, shell paths, configuration references,
  dynamic imports, plugin registrations, packaging, and documentation.
- Preserve compatibility entry points where existing commands or saved
  models depend on them.
- Search for stale references after each migration.
- Revise organization guidance to match the chosen layout, respecting
  current user instructions and preserving project invariants.
- Keep behavioral changes separate from organization changes.

## Handle generated files carefully

- Identify tracked ignored files with:
  git ls-files -ci --exclude-standard
- Inspect candidates before treating them as disposable.
- Preserve useful experiment provenance before untracking artifacts.
- Untrack confirmed generated files using exact paths and retain local
  copies. Adding ignore rules alone does not untrack existing files.
- Do not broadly delete environments, datasets, checkpoints, or results.
- Do not remove code solely because a text search found no callers;
  inspect dynamic loading and external entry points.

## EchoSonarVideo considerations

Verify these against the current branch before acting:

- tool_env contains the tool environment.
- data_core contains data preparation, rewards, serialization, and metrics.
- verl_bridge contains verl integration.
- echoprime_track contains the EchoPrime plus text-model track.
- These responsibilities can guide organization without requiring the
  existing physical layout.
- pyproject.toml explicitly packages echoprime_track and registers a
  vllm.general_plugins entry point. Update packaging and verify plugin
  discovery if moving or introducing nested packages.
- Existing pytest discovery covers tool_env, data_core, and verl_bridge.
  Do not claim EchoPrime coverage from that suite.
- Inspect launchers and Hydra paths before moving configurations.
- Preserve external submodule pins and patch application paths.
- Preserve training partitions, metric definitions, prompt/tool
  contracts, checkpoint compatibility, and operational defaults.
- Read CLAUDE.md, SPEC.md, PLAN.md, and docs/OPEN_ISSUES.md for current
  research and operational context.

## Verify and explain the result

- Run relevant existing tests for code moves.
- Check shell syntax, packaging, configuration paths, and documentation
  links when affected.
- Separate existing failures from cleanup regressions.
- State which model or GPU checks could not be run. Do not launch
  training merely to validate organization.
- Update AGENTS.md with durable placement and naming rules.
- Add or update the root README with a simple folder guide and verified
  entry commands.
- Include a "Where should I put a new file?" table.
- Finish with what changed, where things belong, actual verification,
  and unresolved issues.

Creating or using this skill does not independently authorize publishing,
merging, deleting unique data, or starting experiments.