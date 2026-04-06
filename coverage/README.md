# IsaacLab Coverage Audits

This directory contains periodic test coverage audits for the IsaacLab project.

## Contents

| File | Date | Commit | Summary |
|---|---|---|---|
| [audit-2026-04-06.md](audit-2026-04-06.md) | 2026-04-06 | `f4aa17f` | Initial coverage audit — 128 test files, ~965 test functions across 6 packages |

## How Audits Are Structured

Each audit document covers:

1. **Executive Summary** — high-level findings
2. **Test Infrastructure** — frameworks, CI/CD configuration, execution requirements
3. **Module-by-Module Breakdown** — per-package and per-submodule coverage table
4. **Test Organization** — directory structure and testing patterns
5. **Branch Comparison** — main vs. release branch coverage differences
6. **Risk Assessment** — critical, high, and medium risk untested paths
7. **Recommendations** — prioritized improvement tasks

## Key Findings (as of 2026-04-06)

- All tests require a live NVIDIA Isaac Sim GPU environment — **no offline test tier exists**
- `isaaclab.ui` module has **zero tests**
- `isaaclab_tasks` (391 source files) has only **19 test functions**
- `isaaclab_mimic` datagen pipeline has only **6 test functions** for 49 source files
- CI runs two test jobs per PR (`test-isaaclab-tasks` and `test-general`) on self-hosted GPU runners

## Running Tests

Tests require Isaac Sim. The standard approach:

```bash
# Via the isaaclab.sh helper (requires Isaac Sim installation)
./isaaclab.sh -p -m pytest source/isaaclab/test/ -v

# Or via Docker (matches CI)
docker run --gpus all <isaac-lab-image> bash -c \
  "cd /workspace/isaaclab && /isaac-sim/python.sh -m pytest source/ -v"
```

See `.github/actions/run-tests/action.yml` for the exact CI invocation.
