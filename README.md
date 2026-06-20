# QxBin Logic Implementations (v2)

**By Rupesh Malpani** | pikk.company | QxBin Framework

**QxBin** — Room-temperature personal cubit simulation on classical hardware.

Replaces rigid flat binary with **Binary Probability Matrices**: spatial grids of fractional probabilities. Uses ratios and positive/negative exponents (n, m) to encode "directed contributions" and simulate superposition (the coin still spinning).

No cryogenics. No massive labs. Democratizing quantum-inspired logic for everyone.

## 🚀 QxBin-C — Native C Port (Shipped!)

High-performance, zero-dependency C implementations for edge and cloud.

- **`qxbin_edge.c`** → **QxBin-C Edge** (single personal cubit)
- **`qxbin_cloud.c`** → **QxBin-C Cloud** (ensemble + optimization + CSV export)

**Compile & Run**
```bash
# Edge
gcc -o qxbin-c-edge qxbin_edge.c -lm
./qxbin-c-edge

# Cloud (add OpenMP for parallel)
gcc -fopenmp -o qxbin-c-cloud qxbin_cloud.c -lm
./qxbin-c-cloud
```

Perfect for Pikkstops edge nodes, embedded devices, and high-speed batch workloads.

## Tier 1: Edge / Desktop (`qxbin_edge.py` + `qxbin_edge.c`)
Personal simulator for learning, prototyping, edge devices, uncertain decision systems.

Features:
- Binary Probability Matrix (grid)
- `apply_superposition` with bias + n/m powers
- Probabilistic measurement (collapse)
- Visualization (Python) / clean matrix print (C)

## Tier 2: Cloud / Server (`qxbin_cloud.py` + `qxbin_cloud.c`)
Scalable ensemble of many cubit chains.

Features:
- Parallel evolution (Numba in Py / serial+OpenMP in C)
- Feedback optimization loop
- Aggregate visualization + CSV export (C)
- Easy to GPU-accelerate

## Core QxBin Math
- **Fractional states**: bias**n and (1-bias)**m
- **Probability Matrix**: 2D grid for multi-dimensional state
- **Superposition blend** + **Chain evolution**
- **Measurement**: probabilistic collapse

## Extension Roadmap
1. Analog input (Hall-effect sensors)
2. Physical prototype (magnet grid)
3. GPU / CUDA port
4. Hybrid Qiskit bridge

**QxBin-C** is now shipped and ready. Fork. Extend. Ship faster.

## Links & Context
- Framework details & visuals: https://www.pikk.co.in/chain-test
- YouTube explanations (Rupesh Malpani / pikk.company)
- X: @rupeshmalpani

MIT License | 2026

---

## License

![License](https://img.shields.io/badge/License-Custom%20MIT-blue)

This repository is released under a **custom MIT license** tailored for the QxBin ecosystem by Rupesh Malpani / pikk.company.

**Key terms:**
- **Free** for testing, experimentation, internal organizational use, and building your own software or improvements using your development resources.
- **51% revenue share** with the copyright holders applies when you create and sell a commercial tool, product, or API (whether for commercial customers or personal/end users).
- Enterprise-scale deployments and strategic partnerships are fully **negotiable** — reach out to [@rupeshmalpani](https://x.com/rupeshmalpani).

See the full [LICENSE](LICENSE) file for complete details.

This structure keeps the doors wide open for builders and tinkerers while ensuring the creators pushing the frontier get sustained support when real commercial value is captured at scale.

Part of the pikk-qxbin vision: Democratizing advanced compute. Ship fast. Align incentives for long-term progress.