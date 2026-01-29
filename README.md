# sciml-practice

This repository contains practice code for various scientific machine learning libraries and frameworks, including DeepXDE and others.

## Table of Contents

### 1D Differential Equation PINNs
- [1d_diffeq_pinn/heat_1d_deepxde.ipynb](1d_diffeq_pinn/heat_1d_deepxde.ipynb): Solving an inverse problem for the 1D heat equation using DeepXDE.
- [1d_diffeq_pinn/wave_1d_deepxde.ipynb](1d_diffeq_pinn/wave_1d_deepxde.ipynb): Solving the 1D wave equation using DeepXDE.

### Blood Flow Simulations
- [blood-flow/vessel_geometries.py](blood-flow/vessel_geometries.py): Generates emulated vessel geometry data for DeepONet branch inputs.
- [blood-flow/main.py](blood-flow/main.py): DeepONet model implementation using DeepXDE for neural operators on blood vessel geometries and Navier-Stokes flows.

## Setup

To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

All code in this repository is designed to be run in Jupyter notebooks. Open the `.ipynb` files in Jupyter to execute the examples.
