# QxBin CUDA / GPU Port

**Status**: v1 (CuPy Python) + Native CUDA C++ kernel shipped (June 2026)

This fulfills and extends roadmap item #3: **GPU / CUDA port**.

## What we have now

### 1. `qxbin_cuda.py` (CuPy)
- High-level, drop-in replacement for `QxBinCloud`
- Same exact fractional exponent logic (`bias**n`, `(1-bias)**m`, blend, normalize)
- Runs entirely on GPU via CuPy broadcasting
- Excellent for rapid prototyping, visualization, integration with existing Python code
- Great performance for most use cases (1024+ cubits in real time)

### 2. `qxbin_cuda.cu` (Native CUDA C++)
- Zero Python overhead
- One block per cubit, cooperative threads + warp shuffle reduction
- Same mathematical core as the Python versions
- Standalone compilable demo (`nvcc ... && ./qxbin_cuda`)
- Foundation for production kernels, multi-GPU, and custom bindings

Both versions use the **exact same QxBin math** that makes the "spinning coin" superposition work on classical hardware.

## Quick start

### CuPy version (recommended first)
```bash
pip install cupy-cuda12x   # match your CUDA version
python qxbin_cuda.py
```

### Native kernel
```bash
nvcc -o qxbin_cuda qxbin_cuda.cu -arch=sm_80   # change arch to your GPU
./qxbin_cuda
```

## Why this compounds (diabolical optimism)

We turned a beautiful mathematical hack (Binary Probability Matrices + fractional states) into something that runs at full GPU throughput *today*.

No waiting for logical qubits. No cryogenics. Just clever representation + modern parallel silicon.

This is how we democratize quantum-inspired computing:
- Personal labs on every developer machine
- Edge + cloud scale for Pikk-style systems
- Foundation for richer probabilistic AI (QxGrok direction)

## Immediate next steps

- Python bindings for the native kernel (pybind11 or nanobind)
- Make `QxBinCloud` / `QxBinCUDA` auto-select best backend
- Benchmarks + scaling curves (CPU vs CuPy vs native)
- Multi-GPU support (simple device partitioning)
- Qiskit / CUDA-Q hybrid bridge
- Integration with analog input roadmap item (Hall sensors)

Fork. Extend. Ship.

— Rupesh Malpani | pikk.company | QxBin framework