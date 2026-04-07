# Isaac Lab Main Branch Full Test Suite Audit

**Date:** 2026-04-07
**Auditor:** Automated test runner (Claude)

---

## 1. Repository Information

- **Repo:** https://github.com/tobyj-nvidia/IsaacLab-fleet
- **Cloned to:** `/tmp/isaaclab-main`
- **Git commit:** `43968e075ae` (Add task result file)
- **Git tag:** `v2.3.2-15-g43968e075ae`
- **Branch:** main

---

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.12 |
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish), Kernel 5.15.0-113-generic |
| GPU | NVIDIA L40 (49386 MB VRAM) |
| NVIDIA Driver | 570.158.01 |
| CUDA Version | 12.8 |
| torch | 2.11.0 |
| nvidia-nccl-cu12 | 2.26.2 |
| nvidia-nccl-cu13 | 2.28.9 (installed but not providing the `.so`) |
| isaacsim (metapkg) | 4.5.0.0 |
| isaacsim SimulationApp | 2.4.2 |
| pytest | 8.4.2 |
| pytest-timeout | 2.4.0 |

---

## 3. Critical Blocker: PyTorch / NCCL Incompatibility

**ALL 121 simulation-dependent tests fail with the same root cause:**

```
ImportError: /home/horde/.local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so:
undefined symbol: ncclCommWindowDeregister
```

**Root cause analysis:**
- `torch 2.11.0` requires NCCL symbol `ncclCommWindowDeregister` which was added in **NCCL 2.27+**
- The NCCL library installed at `nvidia/nccl/lib/libnccl.so.2` reports version **2.26.2** (compiled with CUDA 12.2)
- `nvidia-nccl-cu12` 2.26.2 is present at `/home/horde/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2`
- `nvidia-nccl-cu13` 2.28.9 is installed as a pip package but its `libnccl.so.2` points to the same 2.26.2 library
- `libtorch_cuda.so` references `nvidia/nccl/lib/libnccl.so.2` via `ldd` path: `../../nvidia/nccl/lib/libnccl.so.2`
- The system NCCL is also too old: `/usr/lib/x86_64-linux-gnu/libnccl.so.2.23.4`

**This makes PyTorch impossible to import in this environment**, which cascades to all Isaac Lab tests since they all import torch (either directly or through `isaaclab` package imports).

**Affected extensions (during SimulationApp startup):**
- `isaacsim.core.simulation_manager` - fails to load
- `isaacsim.core.cloner` - fails to load
- `isaacsim.core.prims` - fails to load
- `isaacsim.core.api` - fails to load
- `isaaclab_assets` - fails to load
- `isaaclab_tasks` - fails to load

---

## 4. Installation Notes

The top-level `pip install -e .` fails because setuptools finds multiple top-level packages in flat layout. The correct approach (used here) is to install each sub-package separately:

```bash
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_tasks
pip install -e source/isaaclab_rl
```

All sub-packages installed successfully.

---

## 5. Test File Inventory

**Total test files found:** 128

**By directory:**

| Directory | Count | AppLauncher/SimulationApp? |
|-----------|-------|---------------------------|
| `source/isaaclab/test/sim/` | 22 | Yes |
| `source/isaaclab/test/utils/` | 15 | Yes (14) + 1 (torch only) |
| `source/isaaclab/test/sensors/` | 14 | Yes |
| `source/isaaclab/test/envs/` | 12 | Yes |
| `source/isaaclab_tasks/test/` | 9 | Yes |
| `source/isaaclab/test/controllers/` | 7 | Yes |
| `source/isaaclab_rl/test/` | 5 | Yes |
| `source/isaaclab_mimic/test/` | 5 | Yes |
| `source/isaaclab/test/managers/` | 5 | Yes |
| `source/isaaclab/test/assets/` | 5 | Yes |
| `source/isaaclab/test/app/` | 4 | Yes |
| `source/isaaclab/test/devices/` | 3 | Yes |
| `source/isaaclab/test/actuators/` | 3 | Yes |
| `scripts/tools/test/` | 3 | No (pure Python) |
| `source/isaaclab_contrib/test/sensors/` | 2 | Yes |
| `source/isaaclab/test/terrains/` | 2 | Yes |
| `source/isaaclab/test/performance/` | 2 | Yes |
| `source/isaaclab/test/deps/` | 2 | No (1 scipy, 1 torch) |
| `source/isaaclab_tasks/test/benchmarking/` | 1 | Yes (needs `carb`) |
| `source/isaaclab_contrib/test/assets/` | 1 | Yes |
| `source/isaaclab_contrib/test/actuators/` | 1 | Yes |
| `source/isaaclab_assets/test/` | 1 | Yes |
| `source/isaaclab/test/scene/` | 1 | Yes |
| `source/isaaclab/test/markers/` | 1 | Yes |
| `docker/test/` | 1 | No (Docker CLI) |

**Test categories:**
- **Truly safe (no torch, no sim):** 5 files (20 test functions)
- **Torch-only (no sim):** 1 file (`test_torch.py`)
- **SimulationApp/AppLauncher dependent:** 121 files
- **Benchmarking (needs carb, not runnable outside sim):** 1 file

---

## 6. Safe Tests Results (No Torch / No Sim)

**Ran:** `source/isaaclab/test/deps/test_scipy.py`, `scripts/tools/test/test_mp4_to_hdf5.py`, `scripts/tools/test/test_hdf5_to_mp4.py`, `scripts/tools/test/test_cosmos_prompt_gen.py`, `docker/test/test_docker.py`

**Summary: 20 passed, 4 failed**

```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /tmp/isaaclab-main
configfile: pyproject.toml
plugins: hydra-core-1.3.2, asyncio-0.26.0, flaky-3.8.1, timeout-2.4.0, mock-3.15.1, anyio-4.13.0
asyncio: mode=strict, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
timeout: 60.0s
timeout method: signal
timeout func_only: False
collecting ... collected 24 items

source/isaaclab/test/deps/test_scipy.py::test_interpolation PASSED
scripts/tools/test/test_mp4_to_hdf5.py::TestMP4ToHDF5::test_get_frames_from_mp4 PASSED
scripts/tools/test/test_mp4_to_hdf5.py::TestMP4ToHDF5::test_get_frames_from_mp4_resize PASSED
scripts/tools/test/test_mp4_to_hdf5.py::TestMP4ToHDF5::test_process_video_and_demo PASSED
scripts/tools/test/test_mp4_to_hdf5.py::TestMP4ToHDF5::test_main_function PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_get_num_demos PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_rgb PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_segmentation PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_normals PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_shaded_segmentation PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_depth PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_invalid_demo PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_write_demo_to_mp4_invalid_key PASSED
scripts/tools/test/test_hdf5_to_mp4.py::TestHDF5ToMP4::test_main_function PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_generate_prompt_valid_templates PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_generate_prompt_invalid_file PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_generate_prompt_invalid_json PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_main_function_single_prompt PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_main_function_multiple_prompts PASSED
scripts/tools/test/test_cosmos_prompt_gen.py::TestCosmosPromptGen::test_main_function_default_output PASSED
docker/test/test_docker.py::test_docker_profiles[base-] FAILED
docker/test/test_docker.py::test_docker_profiles[base-test] FAILED
docker/test/test_docker.py::test_docker_profiles[ros2-] FAILED
docker/test/test_docker.py::test_docker_profiles[ros2-test] FAILED

========================= 4 failed, 20 passed in 7.28s =========================
```

**Failure reason for docker tests:** Docker is not installed in this environment.
```
RuntimeError: Docker is not installed! Please check the 'Docker Guide' for instruction
```

---

## 7. Torch-only Test Results

`source/isaaclab/test/deps/test_torch.py` — **COLLECTION ERROR**

```
ImportError: /home/horde/.local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so:
undefined symbol: ncclCommWindowDeregister
```

Torch 2.11.0 cannot be imported due to the NCCL symbol mismatch. This test cannot be collected or run.

---

## 8. Simulation-Dependent Tests Results

**121 test files** — all fail with **collection error** (same NCCL/torch issue).

**Sample run** (`source/isaaclab/test/utils/test_math.py`):
```
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
___________ ERROR collecting source/isaaclab/test/utils/test_math.py ___________
ImportError while importing test module '/tmp/isaaclab-main/source/isaaclab/test/utils/test_math.py'.
E   ImportError: /home/horde/.local/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so:
    undefined symbol: ncclCommWindowDeregister
=========================== short test summary info ============================
ERROR source/isaaclab/test/utils/test_math.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 4.08s ==============================
```

The test collection takes ~4 seconds because it still starts the SimulationApp. The SimulationApp starts successfully (takes ~3.5s), loads carb/Vulkan/GPU context, but then fails when trying to load Isaac Lab Python extensions that import torch.

**Extensions that failed to load during SimulationApp startup** (all due to same NCCL/torch error):
- `isaacsim.core.simulation_manager-0.3.3`
- `isaacsim.core.cloner-1.3.4`
- `isaacsim.core.prims-0.3.7`
- `isaacsim.core.api-4.2.16`
- `isaaclab_assets-0.2.4`
- `isaaclab_tasks-0.11.14`

**SimulationApp itself** starts and reaches "app ready" state successfully (GPU/Vulkan/physics foundation all initialize). The failure only happens at the Python extension loading stage.

---

## 9. Benchmarking Tests

`source/isaaclab_tasks/test/benchmarking/test_environments_training.py`

**Status: Cannot run.** The `conftest.py` in that directory imports `carb` directly (from `env_benchmark_test_utils.py`), which requires the full sim context before pytest even starts. Running via system `python3 -m pytest` fails immediately:

```
ModuleNotFoundError: No module named 'carb'
```

---

## 10. Summary Table

| Category | Files | Test Functions | Status |
|----------|-------|---------------|--------|
| No torch / No sim (safe) | 5 | 20 passed, 4 failed | Partial (docker failures expected) |
| Torch-only (`test_torch.py`) | 1 | 5 | BLOCKED by NCCL error |
| SimulationApp/AppLauncher | 121 | Unknown | BLOCKED by NCCL error |
| Benchmarking (needs carb) | 1 | Unknown | BLOCKED (no carb in sys python) |
| **Total** | **128** | **≥ 24 attempted** | **20/24 passed (for runnable tests)** |

---

## 11. Fully Passing Test Directories

| Directory | Status | Notes |
|-----------|--------|-------|
| `scripts/tools/test/` | **FULLY PASSING** | 19/19 tests pass |
| `source/isaaclab/test/deps/` (test_scipy only) | **PASSING** | 1/1 passes; test_torch blocked |
| `docker/test/` | FAILING | Docker not installed |
| All others (121 files) | BLOCKED | NCCL/torch incompatibility |

---

## 12. Known Failure Categories

### Category 1: NCCL/PyTorch Symbol Mismatch (CRITICAL)
- **Error:** `ImportError: libtorch_cuda.so: undefined symbol: ncclCommWindowDeregister`
- **Cause:** `torch 2.11.0` requires NCCL 2.27+, but installed `nvidia-nccl-cu12 2.26.2` provides `ncclCommWindowDeregister` was not added until NCCL 2.27
- **Fix needed:** Upgrade `nvidia-nccl-cu12` to 2.27+ (or ensure `nvidia-nccl-cu13` properly installs a 2.27+ NCCL library)
- **Impact:** 121/128 test files cannot even collect

### Category 2: Docker Not Installed
- **Error:** `RuntimeError: Docker is not installed!`
- **Cause:** Docker CLI not available in this environment
- **Impact:** 4 docker test variants fail
- **Fix:** Expected behavior in non-Docker CI environment; these tests should be skipped with `pytest.skip` if Docker is not available

### Category 3: Carb Module Not in System Python (benchmarking)
- **Error:** `ModuleNotFoundError: No module named 'carb'`
- **Cause:** `carb` is only available inside an active Isaac Sim session; the benchmarking conftest.py imports it at module level
- **Impact:** 1 benchmarking test file cannot be run via system `python3 -m pytest`
- **Fix:** Benchmarking tests must be run through the `isaaclab.sh` launcher or inside SimulationApp context

---

## 13. SimulationApp Behavior Notes

- SimulationApp (headless) starts successfully in ~3.5s and reaches "app ready" state
- GPU (NVIDIA L40) is detected and used: Vulkan, PhysX foundation initialize OK
- Isaac Sim 4.5.0.0 extensions all load without errors at the non-Python level
- Python extension loading is where everything fails (torch import → NCCL error)
- The isaaclab `AppLauncher` wrapper correctly identifies `cuda:0` device before failure
- Warning about omniverse hub inaccessibility (`OmniHub is inaccessible`) — expected in offline environments
- Audio device warnings (`carb.audio.device is misconfigured`) — non-fatal, expected in headless environments

---

## 14. Deprecation Warnings Observed

Multiple Isaac Sim extensions print deprecation warnings during loading:
- `omni.isaac.dynamic_control` → prefer `isaacsim.core`
- `omni.isaac.wheeled_robots` → prefer `isaacsim.robot.wheeled_robots`
- `omni.isaac.franka` → prefer `isaacsim.robot.manipulators.examples.franka`
- `omni.isaac.kit` → prefer `isaacsim.simulation_app`
- `omni.replicator.isaac` → prefer `isaacsim.replicator.*`

These are warnings only and do not cause test failures themselves.

---

## 15. Pytest Output Files

- Safe tests output: `/tmp/isaaclab-safe.txt`
- Sim test sample output: `/tmp/isaaclab-sim.txt`

---

## 16. Recommendations

1. **Fix NCCL version:** The most critical fix is ensuring `nvidia-nccl-cu13 2.28.9` (or any NCCL 2.27+) provides an actual `libnccl.so.2` that includes `ncclCommWindowDeregister`. Currently the pip package `nvidia-nccl-cu13` is installed but its library file at `nvidia/nccl/lib/libnccl.so.2` is still version 2.26.2 from the `nvidia-nccl-cu12` package. Uninstalling `nvidia-nccl-cu12` and reinstalling `nvidia-nccl-cu13` properly may fix this.

2. **Docker test skipping:** The docker tests should check for Docker availability and skip cleanly rather than fail. A `pytest.mark.skipif(not shutil.which('docker'), ...)` would make these environment-appropriate.

3. **Benchmarking conftest:** The benchmarking conftest should use lazy imports or `pytest.importorskip('carb')` to fail gracefully when not running inside Isaac Sim.

4. **Test isolation:** Consider adding a CI category that distinguishes "runs without Isaac Sim" tests from "requires Isaac Sim" tests with explicit marks to make it easier to run the subset that doesn't need the full sim stack.
