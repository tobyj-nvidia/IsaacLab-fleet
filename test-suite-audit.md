# Isaac Lab Test Suite Audit

**Date**: 2026-04-07
**Branch**: fleet/473a4914-47df-4f39-870d-1f6ec4dc6404
**Environment**: Python 3.10.12, PyTorch 2.5.1+cu121, pytest 9.0.2
**CUDA status**: Driver version 12080 (too old for PyTorch 2.5.1 — CUDA unavailable at runtime)

---

## Summary

| Category | Count |
|----------|-------|
| Total test files | 128 |
| Sim-required (AppLauncher) | 121 |
| Truly CPU-safe (no sim deps) | 7 |
| **CPU-safe passed** | **20** |
| CPU-safe failed | 6 |
| CPU-safe skipped | 0 |
| Infrastructure-only (no runnable tests) | 1 |
| Requires Docker | 1 |

---

## CPU-Safe Tests Executed

These 6 test files have no dependency on `AppLauncher`, `SimulationApp`, `omni.*`, `isaacsim.*`, or `pxr` (USD):

### Passed (20/26 tests)

#### `source/isaaclab/test/deps/test_scipy.py` — 1/1 passed
- `test_interpolation` — validates scipy interpolation methods

#### `scripts/tools/test/test_cosmos_prompt_gen.py` — 6/6 passed
- `test_generate_prompt_valid_templates`
- `test_generate_prompt_invalid_file`
- `test_generate_prompt_invalid_json`
- `test_main_function_single_prompt`
- `test_main_function_multiple_prompts`
- `test_main_function_default_output`

#### `scripts/tools/test/test_hdf5_to_mp4.py` — 9/9 passed
- `test_get_num_demos`
- `test_write_demo_to_mp4_rgb`
- `test_write_demo_to_mp4_segmentation`
- `test_write_demo_to_mp4_normals`
- `test_write_demo_to_mp4_shaded_segmentation`
- `test_write_demo_to_mp4_depth`
- `test_write_demo_to_mp4_invalid_demo`
- `test_write_demo_to_mp4_invalid_key`
- `test_main_function`

#### `scripts/tools/test/test_mp4_to_hdf5.py` — 4/4 passed
- `test_get_frames_from_mp4`
- `test_get_frames_from_mp4_resize`
- `test_process_video_and_demo`
- `test_main_function`

### Failed (6/26 tests)

#### `source/isaaclab/test/deps/test_torch.py` — 0/6 passed
All 6 tests fail with `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`.

**Root cause**: These tests use `device="cuda:0"` unconditionally, and PyTorch 2.5.1 requires a newer CUDA driver than what is available on this machine.

Tests affected:
- `test_array_slicing`
- `test_array_circular`
- `test_array_circular_copy`
- `test_array_multi_indexing`
- `test_array_single_indexing`
- `test_logical_or`

**Note**: These tests are marked `@pytest.mark.isaacsim_ci` — they are designed for the full CI environment with CUDA.

---

## Unsafe Tests (Require Simulator)

### Require AppLauncher / Isaac Sim (121 files)

These tests call `AppLauncher(headless=True).app` at module level. Running them without Isaac Sim installed **will hang or crash immediately** — they cannot be collected by pytest safely.

| Directory | File count | Why unsafe |
|-----------|-----------|------------|
| `source/isaaclab/test/actuators/` | 3 | AppLauncher (warp dependency) |
| `source/isaaclab/test/app/` | 4 | AppLauncher (app launch tests) |
| `source/isaaclab/test/assets/` | 5 | AppLauncher (USD physics) |
| `source/isaaclab/test/controllers/` | 7 | AppLauncher (warp/kinematics) |
| `source/isaaclab/test/devices/` | 3 | AppLauncher (retargeting) |
| `source/isaaclab/test/envs/` | 12 | AppLauncher (simulation envs) |
| `source/isaaclab/test/managers/` | 5 | AppLauncher (manager tests) |
| `source/isaaclab/test/markers/` | 1 | AppLauncher (visualization) |
| `source/isaaclab/test/performance/` | 2 | AppLauncher (kit startup) |
| `source/isaaclab/test/scene/` | 1 | AppLauncher (interactive scene) |
| `source/isaaclab/test/sensors/` | 14 | AppLauncher (sensor sim) |
| `source/isaaclab/test/sim/` | 22 | AppLauncher (simulation context) |
| `source/isaaclab/test/terrains/` | 2 | AppLauncher (terrain generation) |
| `source/isaaclab/test/utils/` | 15 | AppLauncher (warp dependency note) |
| `source/isaaclab_assets/test/` | 1 | AppLauncher (asset config validation) |
| `source/isaaclab_contrib/test/` | 4 | AppLauncher (contrib sensors/assets) |
| `source/isaaclab_mimic/test/` | 5 | AppLauncher (mimic/curobo) |
| `source/isaaclab_rl/test/` | 5 | AppLauncher (RL wrapper envs) |
| `source/isaaclab_tasks/test/` | 10 | AppLauncher (task environments) |

**Key insight**: Even tests in `utils/` (math, string, configclass, etc.) use AppLauncher because `isaaclab.utils` imports `pxr` (USD) transitively via `mesh.py`. There is no subset of `isaaclab.*` that can be imported without a running Isaac Sim.

### Requires Docker (1 file)
- `docker/test/test_docker.py` — starts and stops Docker containers; requires Docker daemon and Isaac Lab container images

### Infrastructure only (1 file)
- `tools/test_settings.py` — defines constants and skip lists; contains no test functions

---

## Test Directory Classification

| Directory | Safe to Run (CPU) | Reason |
|-----------|-------------------|--------|
| `source/isaaclab/test/deps/` | **Partial** | scipy: yes; torch: CUDA required |
| `scripts/tools/test/` | **Yes** | Pure Python, no sim deps |
| `source/isaaclab/test/utils/` | **No** | AppLauncher (warp/pxr transitive) |
| `source/isaaclab/test/actuators/` | **No** | AppLauncher |
| `source/isaaclab/test/app/` | **No** | AppLauncher (direct) |
| `source/isaaclab/test/assets/` | **No** | AppLauncher |
| `source/isaaclab/test/controllers/` | **No** | AppLauncher |
| `source/isaaclab/test/devices/` | **No** | AppLauncher |
| `source/isaaclab/test/envs/` | **No** | AppLauncher |
| `source/isaaclab/test/managers/` | **No** | AppLauncher |
| `source/isaaclab/test/markers/` | **No** | AppLauncher |
| `source/isaaclab/test/performance/` | **No** | AppLauncher |
| `source/isaaclab/test/scene/` | **No** | AppLauncher |
| `source/isaaclab/test/sensors/` | **No** | AppLauncher |
| `source/isaaclab/test/sim/` | **No** | AppLauncher |
| `source/isaaclab/test/terrains/` | **No** | AppLauncher |
| `source/isaaclab_assets/test/` | **No** | AppLauncher |
| `source/isaaclab_contrib/test/` | **No** | AppLauncher |
| `source/isaaclab_mimic/test/` | **No** | AppLauncher |
| `source/isaaclab_rl/test/` | **No** | AppLauncher |
| `source/isaaclab_tasks/test/` | **No** | AppLauncher |
| `docker/test/` | **No** | Requires Docker daemon |

---

## Known Failures and Root Causes

### CUDA Driver Too Old (6 failures)
- **Files**: `source/isaaclab/test/deps/test_torch.py`
- **Error**: `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`
- **Root cause**: PyTorch 2.5.1 requires a CUDA driver ≥ 12.4; this system has 12.0.8
- **Fix**: Update NVIDIA driver or use `pytest -m "not isaacsim_ci"` to skip CUDA-dependent tests

### Tests Skipped by tools/test_settings.py
The project maintains a skip list in `tools/test_settings.py` (`TESTS_TO_SKIP`):
- `test_argparser_launch.py` — app.close issue
- `test_build_simulation_context_nonheadless.py` — headless
- `test_env_var_launch.py` — app.close issue
- `test_kwarg_launch.py` — app.close issue
- `test_differential_ik.py` — known failure
- `test_record_video.py` — known failure
- `test_tiled_camera_env.py` — logic improvements needed

---

## How to Run Safe Tests

```bash
# Run only the CPU-safe tests (no simulator required)
cd /path/to/IsaacLab
python3 -m pytest \
  source/isaaclab/test/deps/test_scipy.py \
  scripts/tools/test/test_cosmos_prompt_gen.py \
  scripts/tools/test/test_hdf5_to_mp4.py \
  scripts/tools/test/test_mp4_to_hdf5.py \
  -v --tb=short --timeout=60

# Skip CUDA tests (works even without CUDA)
python3 -m pytest source/isaaclab/test/deps/ -v -m "not isaacsim_ci"
```

## How to Run Full Suite (Requires Isaac Sim)

```bash
# Full suite requires Isaac Sim / Omniverse to be installed
# See: https://isaac-sim.github.io/IsaacLab/
./isaaclab.sh -p -m pytest source/ -v --timeout=300
```

---

## Context from Prior Audits

Based on known project history (v2.3.2):
- 1072/1097 CPU-only tests pass (in environments with warp available)
- 148/1226 full-suite tests fail (Isaac Sim 4.5 API mismatches)
- The discrepancy from this audit is because even "CPU-only" tests in `source/isaaclab/` require AppLauncher for warp initialization
