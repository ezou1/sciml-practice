import deepxde as dde
import numpy as np
import torch
from scipy.interpolate import interp1d
from vessel_discrete_space import VesselDiscreteSpace

# constants
rho = 1.06 # Blood density [g/cm^3]
nu = 0.035 # Kinematic viscosity [cm^2/s] (v in the text)
E = 1e6 # Young's modulus [dyn/cm^2] (approximate for arteries)
h0 = 0.05 # Wall thickness [cm]
p0_ext = 0.0 # External pressure (p0 in text)
delta = 0.1 # Boundary layer thickness (delta in text, approx value)

# define PDE residuals
def pde_residuals(x, y, aux): # x is (x, t), y is predicted q and A, aux is branch function A0, through interpolation
    A = torch.clamp(y[:, 0:1], min=1e-6)  # cross-sectional area
    q = y[:, 1:2]

    A0 = aux
    A0 = torch.clamp(A0, min=1e-6)  # baseline area

    # compute derivatives
    dq_dx = dde.grad.jacobian(y, x, i=1, j=0)
    dA_dt = dde.grad.jacobian(y, x, i=0, j=1)
    dq_dt = dde.grad.jacobian(y, x, i=1, j=1)

    # 1. state equation
    r0 = torch.sqrt(A0 / np.pi)
    beta = (4/3) * (E * h0 / r0)
    p = p0_ext + beta * (1 - torch.sqrt(A0 / A))
    dp_dx = dde.grad.jacobian(p, x, i=0, j=0)

    # 2. continuity equation
    continuity_residual = dq_dx + dA_dt

    # 3. momentum equation
    flux = (q**2) / A
    dflux_dx = dde.grad.jacobian(flux, x, i=0, j=0)
    # friction term
    r = torch.sqrt(A / np.pi)
    friction = -2 * np.pi * nu * r / delta * (q / A)
    momentum_residual = dq_dt + dflux_dx + (A / rho) * dp_dx - friction

    return [continuity_residual, momentum_residual]


# define geometries
geom = dde.geometry.Interval(0, 0.1)
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# pde object (no IC/BCs here, just the operator)
pde_object = dde.data.PDE(geomtime, pde_residuals, [], num_domain=100)

# Load the vessel geometric data
branch_input = np.load("vessel_geometries.npy").astype(np.float32)

# Define trunk input: x and t coordinates
num_samples = branch_input.shape[0]
m_points = branch_input.shape[1]

# Prepare coordinates for deeponet
x_grid = np.linspace(0, 0.1, m_points).astype(np.float32).reshape(-1, 1)  # spatial domain
t_grid = np.linspace(0, 1.0, 100).astype(np.float32).reshape(-1, 1)  # time domain
# trunk is the combination of x and t points
trunk_input = np.vstack([np.append(x, t) for x in x_grid for t in t_grid])

# create eval points for the operator
eval_pts = np.linspace(0, 0.1, m_points).astype(np.float32).reshape(-1, 1) # Sensors locations

# define function space
func_space = VesselDiscreteSpace(branch_input, x_grid.flatten())


# pde operator
pde_op = dde.data.PDEOperatorCartesianProd(
    pde_object,
    func_space, # branch function space
    eval_pts,           # eval points
    num_function=1000,   # Matches your num_samples
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
model.compile(
    optimizer="adam", 
    lr=0.001, 
)

# train the model
losshistory, train_state = model.train(iterations=500, display_every=100) # default 20000, 1000
model.save("blood_flow_deeponet")
