# QxBin CUDA / GPU Port

**Status**: Full stack shipped — CuPy Python + Native CUDA C++ kernel + Python bindings (June 2026)

This fulfills roadmap item #3 and goes further: we now have a complete, layered GPU implementation of QxBin.

## Current stack (pick what you need)

| Layer                  | File                    | When to use                              | Performance     | Ease of use |
|------------------------|-------------------------|------------------------------------------|-----------------|-------------|
| **Python (high-level)**| `qxbin_cuda.py`        | Prototyping, viz, quick experiments     | Very good       | Excellent   |
| **Native CUDA kernel** | `qxbin_cuda.cu`        | Maximum speed, production, custom work  | Best            | Medium      |
| **Python bindings**    | `qxbin_cuda_pybind.cpp` + `setup.py` | Best of both worlds (speed + Python)   | Excellent       | Excellent   |

All three layers use the **exact same QxBin fractional superposition math**.

## Quick start (recommended path)

### 1. High-level CuPy (fastest to try)
```bash
pip install cupy-cuda12x
python qxbin_cuda.py
```

### 2. Native + Python bindings (recommended for real work)
```bash
pip install pybind11 numpy

# Build the binding
python setup.py build_ext --inplace

# Or manual compile if you prefer
c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
    qxbin_cuda_pybind.cpp -o qxbin_cuda$(python3-config --extension-suffix) \
    -L/usr/local/cuda/lib64 -lcudart

python -c "from qxbin_cuda import QxBinNative; qx = QxBinNative(1024, 8); print(qx); qx.optimize_to_target(0.72)"
```

## What this unlocks right now

- Run 1024+ evolving cubit ensembles at interactive speeds on a single GPU
- Use the same probabilistic logic in edge, cloud, or research workflows
- Foundation for QxGrok-style probabilistic MoE, uncertainty-aware systems, and quantum-inspired optimization
- Clean path to future hybrid Qiskit + native CUDA workflows

## Next (pick your priority)

- Auto backend selection (unified `QxBin` class that picks best available)
- Benchmarks & scaling curves
- Multi-GPU support
- Qiskit / CUDA-Q bridge
- Integration with Pikk edge infrastructure
- Analog input (Hall-effect sensors) prototype

We are shipping fast because the math is solid and the hardware is already here.

Fork it. Extend it. Ship faster.

— Rupesh Malpani | pikk.company | QxBin framework