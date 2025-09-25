# TFL Demo

This repo demonstrates a simple, formal, **STL-based** time–frequency anomaly detector:
- Build **z_out(t)** by aggregating STFT energy **outside** user-defined *safe* bands.
- Monitor an STL spec `G_[0,T](zout[t] <= th)` with **Breach**.
- Learn the scalar threshold **θ** by **minimizing training MCR** via MATLAB's `simulannealbnd`.
- Evaluate on a held-out test set; plot results.

## Requirements
- MATLAB (R2019b+ recommended)
- Global Optimization Toolbox (for `simulannealbnd`)
- **Breach** toolbox on the MATLAB path (e.g., `addpath(genpath('<path-to-breach>'))`)

## Quick start
```matlab
cd tfl_demo
run_demo
```