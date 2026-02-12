import deepxde as dde
import numpy as np
import torch
from scipy.interpolate import interp1d
from vessel_discrete_space import VesselDiscreteSpace

dde.config.set_default_float("float64")

# constants
rho = 1.06 # Blood density [g/cm^3]
nu = 0.035 # Kinematic viscosity [cm^2/s] (v in the text)
E = 1e6 # Young's modulus [dyn/cm^2] (approximate for arteries)
h0 = 0.05 # Wall thickness [cm]
p0_ext = 0.0 # External pressure (p0 in text)
delta = 0.1 # Boundary layer thickness (delta in text, approx value)

# --- Non-dimensionalization (all constants in CGS: cm, g, s, dyn) ---
L_ref = 10.0  # characteristic length [cm] (vessel segment ~10 cm)
A_ref = 0.8   # characteristic area [cm^2] (baseline cross-section)

# Compute the Moens-Korteweg wave speed as the natural velocity scale.
# This is the standard choice for 1D blood flow and ensures the
# non-dimensional pressure and beta are O(1).
r0_ref = np.sqrt(A_ref / np.pi)                # reference radius [cm]
beta_0 = (4.0 / 3.0) * E * h0 / r0_ref         # tube law coefficient [dyn/cm^2]
c_0 = np.sqrt(beta_0 / (2.0 * rho))             # pulse wave speed [cm/s]
u_ref = c_0                                      # ~250 cm/s

# Derived scales
t_ref = L_ref / u_ref          # time scale [s]  (~0.04 s)
p_ref = rho * (u_ref ** 2)     # pressure scale   (rho * c0^2)
q_ref = A_ref * u_ref          # flow rate scale  [cm^3/s]

# Non-dimensional constants
E_hat = E / p_ref              # ~15   (was ~94 with old u_ref)
h0_hat = h0 / L_ref            # 0.005 (was 0.5 with L_ref in wrong units)

# Non-dimensional friction group: (2*pi*nu*L_ref) / (u_ref * delta)
friction_coeff = (2 * np.pi * nu * L_ref) / (u_ref * delta)

print(f"Scaling: u_ref={u_ref:.1f} cm/s, t_ref={t_ref:.4f} s, "
      f"E_hat={E_hat:.2f}, h0_hat={h0_hat:.4f}, friction={friction_coeff:.4f}")
print(f"beta_hat(A0=A_ref) = {(4/3)*E_hat*h0_hat/np.sqrt(1/np.pi):.3f}  (should be ~O(1))")

# --- NEW: Physiological Constants for BCs ---
HR = 60.0    # Heart Rate (beats per min)
T_cycle = 60.0 / HR # Duration of one cardiac cycle
q_mean_val = 2.0 # Mean flow rate (approx 2 ml/s for small artery)
# Normalized mean flow
q_hat_mean = q_mean_val / q_ref 

# Resistance for outflow (Simple Resistive BC)
# Real Windkessel is complex; we start with P = R*Q for stability
R_out = 0.5 # Dimensionless resistance


# define PDE residuals
def pde_residuals(x, y, aux): # x is (x, t), y is predicted q and A, aux is branch function A0, through interpolation
    A_hat = torch.clamp(y[:, 0:1], min=1e-6)  # non-dimensional cross-sectional area
    q_hat = y[:, 1:2]  # non-dimensional flow rate

    # aux comes from the function space (raw A0 in cm^2), normalize it here
    A0_hat = aux / A_ref  # baseline area from branch function space, now non-dimensional

    # compute derivatives (all w.r.t. non-dimensional x_hat, t_hat)
    dq_dxh = dde.grad.jacobian(y, x, i=1, j=0)
    dA_dth = dde.grad.jacobian(y, x, i=0, j=1)
    dq_dth = dde.grad.jacobian(y, x, i=1, j=1)

    # State equation using SCALED constants
    r0_hat = torch.sqrt(A0_hat / np.pi)
    # beta_hat is dimensionless: (4/3) * (E/p_ref) * (h0/L_ref) / r0_hat
    beta_hat = (4.0 / 3.0) * (E_hat * h0_hat / r0_hat)
    p_hat = beta_hat * (1.0 - torch.sqrt(A0_hat / A_hat))
    dp_dxh = dde.grad.jacobian(p_hat, x, i=0, j=0)

    # 1. Continuity equation (non-dimensional):
    #    dA_hat/dt_hat + dq_hat/dx_hat = 0
    continuity_res = dq_dxh + dA_dth

    # 2. Momentum equation (non-dimensional):
    #    dq_hat/dt_hat + d(q_hat^2/A_hat)/dx_hat + A_hat * dp_hat/dx_hat = friction
    #    Note: rho is already absorbed into p_ref = rho * u_ref^2, so the
    #    pressure gradient term is just A_hat * dp_hat/dx_hat (no rho division).
    flux = (q_hat ** 2) / A_hat
    dflux_dx = dde.grad.jacobian(flux, x, i=0, j=0)

    # Friction term (non-dimensional):
    #   friction_coeff * (r_hat * q_hat / A_hat)
    #   where friction_coeff = 2*pi*nu*L_ref / (u_ref*delta)
    r_hat = torch.sqrt(A_hat / np.pi)
    friction = -friction_coeff * (r_hat * q_hat / A_hat)

    momentum_residual = dq_dth + dflux_dx + A_hat * dp_dxh - friction

    return [continuity_res, momentum_residual]


# define geometries
geom = dde.geometry.Interval(0, 1.0)
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# IC/BC for the PDE
# Inflow BC is the heartbeat, outflow bc is just resistance, IC vessel starts at equilibrium (q=0, A=A0)

# Inflow BC (Heartbeat) - Eq 7 in Appendix
def inflow_func(x):
    # x[:, 1] is time 't'
    t = x[:, 1:2]
    # Period T in dimensionless time
    # We need to convert physical time T_cycle to dimensionless time
    T_dim = T_cycle / t_ref 
    
    # Eq 7: q = q_mean * (sin(...) + 1)
    # Note: We use dimensionless t and T
    omega = 2 * np.pi / T_dim
    val = q_hat_mean * (np.sin(omega * t - np.pi/2) + 1)
    return val


# Initial Condition for Flow (Start at rest)
def ic_q_func(x):
    return np.zeros_like(x[:, 0:1])


# Initial Condition for Area (Start at A0)
# This requires an OperatorBC because it depends on the Branch Input (A0)
def ic_A_residual(x, y, aux):
    if not torch.is_tensor(aux):
        aux = torch.tensor(aux, dtype=y.dtype, device=y.device)
    
    # y[:, 0] is A_hat (non-dimensional)
    A_hat = torch.clamp(y[:, 0:1], min=1e-6)
    # aux is the raw A0 from function space (dimensional, cm^2)
    A0_hat = aux / A_ref  # normalize to match A_hat's scale
    
    # IC: A_hat = A0_hat at t=0
    return A_hat - A0_hat


# 4. Outflow BC (Simple Resistance)
# P_out = R * Q_out (Dimensionless)
def outflow_residual(x, y, aux):
    if not torch.is_tensor(aux):
        aux = torch.tensor(aux, dtype=y.dtype, device=y.device)
    
    A_hat = torch.clamp(y[:, 0:1], min=1e-6)
    q_hat = y[:, 1:2]
    # aux is raw A0 (dimensional), normalize once
    A0_hat = aux / A_ref

    # Reuse global E_hat, h0_hat (already non-dimensionalized)
    r0_hat = torch.sqrt(A0_hat / np.pi)
    beta_hat = (4.0 / 3.0) * (E_hat * h0_hat / r0_hat)
    p_hat = beta_hat * (1.0 - torch.sqrt(A0_hat / A_hat))
    
    # Resistive BC: p_hat - R_out * q_hat = 0
    return p_hat - (R_out * q_hat)


# DEFINE all the BC/IC objects
bc_inflow = dde.icbc.DirichletBC(
    geomtime, 
    inflow_func, 
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0),
    component=1 # component 1 is 'q'
)
bc_outflow = dde.icbc.OperatorBC(
    geomtime,
    outflow_residual,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 1.0)
)
ic_q = dde.icbc.IC(
    geomtime,
    ic_q_func,
    lambda x, on_initial: on_initial,
    component=1 # component 1 is 'q'
)
ic_A = dde.icbc.OperatorBC(
    geomtime,
    ic_A_residual,
    lambda x, on_initial: on_initial
)

# pde object for the operator
pde_object = dde.data.TimePDE(
    geomtime, 
    pde_residuals, 
    [bc_inflow, bc_outflow, ic_q, ic_A],
    num_domain=500,
    num_boundary=100,   # points on x=0 and x=1 (for inflow/outflow BCs)
    num_initial=100     # points at t=0 (for initial conditions)
)

# Load the vessel geometric data (raw, dimensional A0 in cm^2)
# Do NOT pre-normalize here — normalization happens inside the PDE/BC residuals
branch_input = np.load("vessel_geometries.npy").astype(np.float64)

# Define trunk input: x and t coordinates
num_samples = branch_input.shape[0]
m_points = branch_input.shape[1]

# Prepare coordinates for deeponet
x_grid = np.linspace(0, 1.0, m_points).astype(np.float64).reshape(-1, 1)  # spatial domain

# create eval points for the operator (sensor locations along the vessel)
eval_pts = np.linspace(0, 1.0, m_points).astype(np.float64).reshape(-1, 1)

# define function space
func_space = VesselDiscreteSpace(branch_input, x_grid.flatten())


# pde operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde_object,
    func_space, # branch function space
    eval_pts, # eval points
    num_function=1000, # Matches your num_samples
    num_test=100,
    batch_size=32 # Start small to speed up each iteration
)


# define DeepONet
net = dde.nn.DeepONetCartesianProd(
    [m_points, 128, 128], # Branch net architecture (Input dim = m_points)
    [2, 128, 128],        # Trunk net architecture (Input dim = 2 for x,t)
    "tanh",               # Activation
    "Glorot normal",       # Initializer
    num_outputs=2,
    multi_output_strategy="independent"
)

# define model
model = dde.Model(pde_op, net)
# Losses: [continuity, momentum, bc_inflow, bc_outflow, ic_q, ic_A]
# Weight BCs/ICs higher so the network learns physical constraints first
model.compile(
    optimizer="adam", 
    lr=1e-3,
    loss_weights=[1, 1, 10, 10, 10, 10] # loss weights to ensure compliance with BC/IC early training
)

# train the model
losshistory, train_state = model.train(iterations=20000, display_every=1000) # default 20000, 1000
model.save("blood_flow_deeponet")