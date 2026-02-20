"""
test.py — Run the trained blood flow DeepONet over all 1000 vessel geometries
            and save predictions (A, q in physical units) to an .npz file.

Usage:
    python test.py
"""

import deepxde as dde
import numpy as np
import torch
import torch.nn.functional as F
from vessel_discrete_space import VesselDiscreteSpace


# Same custom BC as main.py — defined here to avoid importing main.py
# (which would trigger its top-level training code).
class AuxOperatorBC(dde.icbc.OperatorBC):
    """OperatorBC that passes auxiliary variables (branch function values)
    instead of raw spatial coordinates to the user-provided function."""

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return self.func(inputs, outputs, aux_var)[beg:end]

dde.config.set_default_float("float64")

# ---- Reference scales (must match main.py exactly) ----
rho = 1.06; E = 1e6; h0 = 0.05; nu = 0.035; delta = 0.1
L_ref = 10.0; A_ref = 0.8
r0_ref = np.sqrt(A_ref / np.pi)
beta_0 = (4.0 / 3.0) * E * h0 / r0_ref
c_0 = np.sqrt(beta_0 / (2.0 * rho))
u_ref = c_0
t_ref = L_ref / u_ref
q_ref = A_ref * u_ref
p_ref = rho * u_ref ** 2
E_hat = E / p_ref
h0_hat = h0 / L_ref
friction_coeff = (2 * np.pi * nu * L_ref) / (u_ref * delta)
HR = 60.0; T_cycle = 60.0 / HR
q_mean_val = 2.0; q_hat_mean = q_mean_val / q_ref; R_out = 0.5

# ---- Dummy PDE / BC setup (required to build the model for restore) ----
geom = dde.geometry.Interval(0, 1.0)
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde_residuals(x, y, aux):
    return [torch.zeros_like(y[:, 0:1]), torch.zeros_like(y[:, 0:1])]

def inflow_func(x):
    t = x[:, 1:2]
    T_dim = T_cycle / t_ref
    omega = 2 * np.pi / T_dim
    return q_hat_mean * (np.sin(omega * t - np.pi / 2) + 1)

def ic_q_func(x):
    return np.zeros_like(x[:, 0:1])

def ic_A_residual(x, y, aux):
    if not torch.is_tensor(aux):
        aux = torch.tensor(aux, dtype=y.dtype, device=y.device)
    return F.softplus(y[:, 0:1]) + 0.1 - aux / A_ref

def outflow_residual(x, y, aux):
    if not torch.is_tensor(aux):
        aux = torch.tensor(aux, dtype=y.dtype, device=y.device)
    A_hat = F.softplus(y[:, 0:1]) + 0.1
    q_hat = y[:, 1:2]
    A0_hat = aux / A_ref
    r0_hat = torch.sqrt(A0_hat / np.pi)
    beta_hat = (4.0 / 3.0) * (E_hat * h0_hat / r0_hat)
    p_hat = beta_hat * (1.0 - torch.sqrt(A0_hat / A_hat))
    return p_hat - R_out * q_hat

bc_inflow = dde.icbc.DirichletBC(geomtime, inflow_func,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0), component=1)
bc_outflow = AuxOperatorBC(geomtime, outflow_residual,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0))
ic_q = dde.icbc.IC(geomtime, ic_q_func,
    lambda x, on_initial: on_initial, component=1)
ic_A = AuxOperatorBC(geomtime, ic_A_residual,
    lambda x, on_initial: on_initial)

pde_object = dde.data.TimePDE(geomtime, pde_residuals,
    [bc_inflow, bc_outflow, ic_q, ic_A],
    num_domain=500, num_boundary=100, num_initial=100)

# ---- Load vessel geometries ----
branch_input = np.load("vessel_geometries.npy").astype(np.float64)
num_vessels = branch_input.shape[0]  # 1000
m_points = branch_input.shape[1]     # 100

x_grid = np.linspace(0, 1.0, m_points, dtype=np.float64).reshape(-1, 1)
eval_pts = np.linspace(0, 1.0, m_points, dtype=np.float64).reshape(-1, 1)
func_space = VesselDiscreteSpace(branch_input, x_grid.flatten())

pde_op = dde.data.PDEOperatorCartesianProd(
    pde_object, func_space, eval_pts,
    num_function=num_vessels, num_test=100, batch_size=32)

# ---- Build network (same architecture as main.py) ----
net = dde.nn.DeepONetCartesianProd(
    [m_points, 128, 128], [2, 128, 128],
    "tanh", "Glorot normal",
    num_outputs=2, multi_output_strategy="independent")

model = dde.Model(pde_op, net)
model.compile(optimizer="adam", lr=1e-4)

# ---- Restore trained weights ----
MODEL_PATH = "blood_flow_deeponet-500.pt"
model.restore(MODEL_PATH, verbose=1)
net.eval()

# ---- Define evaluation grid ----
N_x = 200   # spatial points
N_t = 300   # time steps
x_pts = np.linspace(0, 1.0, N_x, dtype=np.float64)
t_pts = np.linspace(0, 1.0, N_t, dtype=np.float64)
xx, tt = np.meshgrid(x_pts, t_pts)
trunk_pts = np.column_stack([xx.ravel(), tt.ravel()])  # (N_x*N_t, 2)
print(f"Evaluation grid: {N_x} x-points, {N_t} t-points = {trunk_pts.shape[0]} total trunk points")

# ---- Run inference in batches over all vessels ----
# Output shapes: (num_vessels, N_t, N_x) for both A and q
BATCH = 50  # vessels per batch (keep memory reasonable)

A_all = np.zeros((num_vessels, N_t, N_x), dtype=np.float64)
q_all = np.zeros((num_vessels, N_t, N_x), dtype=np.float64)

print(f"Running inference on {num_vessels} vessels in batches of {BATCH}...")

with torch.no_grad():
    trunk_tensor = torch.as_tensor(trunk_pts, dtype=torch.float64)

    for start in range(0, num_vessels, BATCH):
        end = min(start + BATCH, num_vessels)
        batch_branch = torch.as_tensor(branch_input[start:end], dtype=torch.float64)

        # DeepONet forward: (batch_size, N_trunk, num_outputs)
        raw_out = net((batch_branch, trunk_tensor))  # (B, N_x*N_t, 2)

        # Apply the same softplus transform used during training
        raw_A = raw_out[:, :, 0]   # (B, N_x*N_t)
        raw_q = raw_out[:, :, 1]   # (B, N_x*N_t)

        A_hat = F.softplus(raw_A) + 0.1  # non-dimensional area (always > 0.1)
        q_hat = raw_q                     # non-dimensional flow rate

        # Convert to physical units
        A_phys = A_hat.numpy() * A_ref    # cm²
        q_phys = q_hat.numpy() * q_ref    # cm³/s

        # Reshape from flat trunk to (N_t, N_x) grid
        B = end - start
        A_all[start:end] = A_phys.reshape(B, N_t, N_x)
        q_all[start:end] = q_phys.reshape(B, N_t, N_x)

        print(f"  vessels {start:4d}–{end-1:4d} done  "
              f"(A range: [{A_phys.min():.4f}, {A_phys.max():.4f}] cm²,  "
              f"q range: [{q_phys.min():.4f}, {q_phys.max():.4f}] cm³/s)")

# ---- Save results ----
OUT_FILE = "blood_flow_results.npz"
np.savez(OUT_FILE,
    # Grid
    x=x_pts * L_ref,               # spatial grid in cm (200,)
    t=t_pts * t_ref,               # time grid in seconds (300,)
    # Predictions — physical units
    A=A_all,                        # area in cm², shape (1000, 300, 200)
    q=q_all,                        # flow in cm³/s, shape (1000, 300, 200)
    # Vessel geometries for reference
    A0=branch_input,                # resting area profiles, (1000, 100)
    # Reference scales (for reproducibility)
    L_ref=L_ref, A_ref=A_ref, t_ref=t_ref, q_ref=q_ref, u_ref=u_ref, p_ref=p_ref,
    model_path=MODEL_PATH,
)

file_size_mb = np.prod(A_all.shape) * 8 * 2 / 1e6  # float64, two arrays
print(f"\nSaved to {OUT_FILE}")
print(f"  Vessels: {num_vessels}")
print(f"  Grid: {N_x} x-points × {N_t} t-steps")
print(f"  Arrays: A {A_all.shape}, q {q_all.shape}")
print(f"  Approx size: {file_size_mb:.0f} MB")
