import deepxde as dde
import numpy as np

# Load the vessel geometric data
branch_input = np.load("vessel_geometries.npy")

# Define trunk input: x and t coordinates
num_samples = branch_input.shape[0]
m_points = branch_input.shape[1]

# Prepare coordinates for deeponet
x_grid = np.linspace(0, 0.1, m_points).reshape(-1, 1)  # spatial domain
t_grid = np.linspace(0, 1.0, 100).reshape(-1, 1)  # time domain
# trunk is the combination of x and t points
trunk_input = np.vstack([np.append(x, t) for x in x_grid for t in t_grid])

# define the DeepONet model
net = dde.nn.DeepONet(
    [m_points, 128, 128],  # branch input shape
    [2, 128, 128],         # trunk input shape (2 because x and t)
    "tanh",
    "Glorot normal"
)

# define PDE residual
