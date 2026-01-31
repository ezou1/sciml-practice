import deepxde as dde
import numpy as np
from scipy.interpolate import interp1d

class VesselDiscreteSpace(dde.data.function_spaces.FunctionSpace):
    def __init__(self, samples, grid_points):
        """
        samples: numpy array of shape (N, m) containing A0 profiles.
        grid_points: numpy array of shape (m,) containing the x-coordinates 
                     corresponding to the columns of 'samples'.
        """
        self.samples = samples
        self.grid_points = grid_points
        self.n_samples = samples.shape[0]

    def random(self, size):
        """
        Randomly selects 'size' vessel profiles from the dataset.
        Returns: numpy array of shape (size, m)
        """
        # Select random indices without replacement
        idx = np.random.choice(self.n_samples, size, replace=False)
        return self.samples[idx]

    def eval_one(self, feature, x):
        """
        Evaluate one function at one point.
        feature: shape (m,) - One A0 profile
        x: shape (dim,) - One coordinate (x, t) in the trunk domain
        
        Returns: float - The interpolated A0 value
        """
        # Extract the spatial coordinate x (index 0). 
        # We ignore t (index 1) because A0 is only space-dependent.
        x_coord = x[0]
        
        # Create interpolation function for this specific profile
        # 'cubic' ensures we have smooth derivatives for the PDE
        f = interp1d(self.grid_points, feature, kind='cubic', 
                     bounds_error=False, fill_value="extrapolate")
        
        return float(f(x_coord))

    def eval_batch(self, features, xs):
        """
        Evaluate a batch of functions at a batch of points.
        features: shape (N, m) - Batch of A0 profiles
        xs: shape (P, dim) - Batch of trunk coordinates (x, t)
        
        Returns: shape (N, P) - The interpolated A0 values
        """
        # Extract x coordinates from the trunk input
        x_eval = xs[:, 0]
        
        # Interpolate all features onto the evaluation points
        f = interp1d(self.grid_points, features, kind='cubic', axis=1, 
                     bounds_error=False, fill_value="extrapolate")
        
        return f(x_eval).astype(np.float32)