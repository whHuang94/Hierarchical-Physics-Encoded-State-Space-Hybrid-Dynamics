# Hierarchical Physics-Encoded State-Space Hybrid Dynamics for Contact Force Sensing of Serial Manipulators Using Intrinsic Sensors

Reference code and processed data for the manuscript submitted to **Mechatronics**
(Manuscript ID: **MECH-D-26-01142**).

This repository currently contains the minimal reproducibility package for the
cross-workload ablation:

```text
.
├── README.md
├── ablation_cross_workload_fixed.py
└── data/
    ├── dataset1.csv
    └── dataset2.csv
```

## Overview

The method decomposes the joint-torque residual of a serial manipulator into

- a gravity branch with sine/cosine input bases;
- a Coriolis branch with quadratic velocity products;
- a unified dissipative branch with two internal states and a shared gate,
  representing friction and hysteresis jointly.

The ablation script compares the full model with four variants:

- **C1**: hysteresis represented by a GRU or a static MLP;
- **C2**: separate LuGre and Bouc-Wen branches;
- **C3**: shared input without structural isolation;
- **C4**: window-independent training without state carry-over.

## Data

The script reads two processed CSV files:

```text
data/dataset1.csv   # source domain, no end load
data/dataset2.csv   # target domain, with end load
```

Each file must contain the following columns:

```text
q1 ... q6              joint positions (rad)
dq1 ... dq6            joint velocities (rad/s)
tau_res1 ... tau_res6  residual torque in current-equivalent units
```

Dataset 1 contains 29,967 samples. Dataset 2 contains 2,800 samples: 800 are used
for adaptation and 2,000 are held out for testing.

The script uses `data/dataset1.csv` and `data/dataset2.csv` by default. To use
another directory or file name, set the environment variables:

```bash
# Windows
set DATA_DIR=D:\path\to\data
set SRC_FILE=dataset1.csv
set TGT_FILE=dataset2.csv

# Linux / macOS
export DATA_DIR=/path/to/data
export SRC_FILE=dataset1.csv
export TGT_FILE=dataset2.csv
```

## Requirements

- Python 3.10+
- NumPy
- pandas
- PyTorch
- matplotlib

Install with:

```bash
pip install numpy pandas torch matplotlib
```

A CUDA-capable GPU is recommended but not required. The script automatically
uses CUDA when available.

## Run

From the repository root:

```bash
python ablation_cross_workload_fixed.py
```

The script prints the source-domain and target-domain MSE for each configuration,
the degradation ratio, and writes:

```text
cross_workload_innovation_ablation.png
```

## Implementation notes

- Training uses 100 epochs, Adam with an initial learning rate of `1e-3`, cosine
  annealing, gradient clipping at `1.0`, batch size 64, segments of 200 samples,
  and chunks of 20 samples.
- Normalization statistics are computed from the training split of Dataset 1 only
  and applied unchanged to the target domain.
- The random seed is fixed to 0.
- The C2 branch advances both the hysteresis state and the friction state at every
  time step.

## Citation

If you use this code or data, please cite the manuscript:

```text
W. Huang, Y. Lin, W. Gan, H. Min, Hierarchical Physics-Encoded State-Space
Hybrid Dynamics for Contact Force Sensing of Serial Manipulators Using
Intrinsic Sensors, Mechatronics (2026), manuscript MECH-D-26-01142.
```

## Contact

- Wenhui Huang: hwenhui@wust.edu.cn
- Huasong Min: mhuasong@wust.edu.cn

Wuhan University of Science and Technology, Wuhan 430080, Hubei, China.
