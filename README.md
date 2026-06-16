# QxBin Logic Implementations (v2)

**By Rupesh Malpani** | pikk.company

**QxBin** — Room-temperature personal cubit simulation on classical hardware.

Replaces rigid flat binary with **Binary Probability Matrices**: spatial grids of fractional probabilities. Uses ratios and positive/negative exponents (n, m) to encode "directed contributions" and simulate superposition (the coin still spinning).

No cryogenics. No massive labs. Just Python you can run on a laptop or scale in the cloud. Democratizing quantum-inspired logic for everyone.

## What's New in v2
- Tighter integration of fractional exponents and coordinate-style probability construction
- More stable evolution and normalization in both tiers
- Clearer mapping to QxBin concepts (probability matrix, coin-toss chains, analog nuance)
- Better comments and extensibility

## Tier 1: Edge / Desktop (`qxbin_edge.py`)
Personal simulator for learning, prototyping, edge devices, uncertain decision systems.

Features:
- Binary Probability Matrix (grid)
- apply_superposition with bias + n/m powers
- Probabilistic measurement (collapse)
- Visualization with node clusters (inspired by the colorful grid/sphere visuals)

**Quick start**
```bash
pip install -r requirements.txt
python qxbin_edge.py
```

## Tier 2: Cloud / Server (`qxbin_cloud.py`)
Scalable ensemble of many cubit chains.

Features:
- Numba-parallel evolution of probability matrices
- Feedback optimization loop
- Aggregate visualization
- Easy to GPU-accelerate or distribute

**Quick start**
```bash
pip install -r requirements.txt
python qxbin_cloud.py
```

## Core QxBin Math in These Implementations
- **Fractional states**: Instead of 0 or 1, we use continuous probabilities (bias ** power)
- **Exponents n & m**: Control the "lean" and direction of contributions in the coordinate system
- **Probability Matrix**: 2D grid representing multi-dimensional state (punch-card analog)
- **Superposition blend**: Combine current matrix with new probabilistic input
- **Chain evolution**: Parallel update across many independent or interacting cubits
- **Measurement**: Weighted random collapse, preserving probability interpretation

## Extension Roadmap (First Principles)
1. Analog input stub: Simulate or connect Hall-effect sensor / pressure data as bias input
2. Physical prototype: Map matrix to LED grid or magnet array + punch card
3. q-binomial / lattice path component: Add explicit chain counting or binomial coefficients
4. Simple "entanglement": Correlate two matrices via shared factors
5. GPU version: Port matrix ops to CuPy / CUDA
6. Hybrid: Bridge to Qiskit for real quantum backend testing
7. MoodBin: Emotional probability clouds on top of the same grid

## Why QxBin Matters
Traditional quantum computing is locked behind expensive, complex infrastructure.
QxBin brings the **thinking** — superposition, probability amplitudes, multi-state logic — to room temperature classical machines.

Students, founders, edge developers, and micro-enterprises can now experiment, prototype, and build with it today.

This is acceleration through accessibility.

Fork. Extend. Ship. Iterate in public.

## Links & Context
- Framework details & visuals: https://www.pikk.co.in/chain-test
- YouTube explanations (Rupesh Malpani / pikk.company)
- X: @rupeshmalpani

MIT License | 2026
