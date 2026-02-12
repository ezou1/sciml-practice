import numpy as np
import torch

# this file defines vessel geometries for blood flow simulations and prepares them as branch input for DeepONet
def generate_vessel_geometries(num_samples, m_points=100, length=0.1):
    #docs
    x = np.linspace(0, length, m_points)
    geometries = []

    for i in range(num_samples):
        # randomly generate vessel cross section profile
        base_area = 0.8 # units??? cm^2, note also that this corresponds with A0 in main.py
        A0 = np.ones_like(x) * base_area # baseline area at equilibrium

        # add stenosis narrowing
        if np.random.rand() > 0.5:
            # randomize location and severity of stenosis
            loc = np.random.uniform(0.2*length, 0.8*length)
            width = np.random.uniform(0.005, 0.015)
            severity = np.random.uniform(0.3, 0.7) # fraction of area reduction

            # use a gaussian dip to simulate narrowing geometry
            A0 = A0 * (1 - severity * np.exp(-((x - loc)**2) / (2 * (width**2))))
        
        geometries.append(A0)
    return np.array(geometries)  # shape (num_samples, m_points)

# generates 1000 vessel geometries
# saves them as .npy file for later entry into DeepONet
branch_inputs = generate_vessel_geometries(1000)
np.save('vessel_geometries.npy', branch_inputs)