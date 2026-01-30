import numpy as np
import matplotlib.pyplot as plt

def visualize_vessel(vessel_path, num_samples=1):
    """
    Loads vessel geometry data (A0 profiles) and plots a side view of the artery.
    Assumes circular cross-sections to derive radius from Area.
    """
    try:
        geometries = np.load(vessel_path)
    except Exception as e:
        print(f"Error loading vessel geometries: {e}")
        return
    
    m_points = geometries.shape[1]
    x_grid = np.linspace(0, 0.1, m_points)

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples), sharex=True)
    if num_samples == 1: axes = [axes] # Handle single sample case

    # display random indicies
    indices = np.random.choice(geometries.shape[0], num_samples, replace=False)
    # indices = [199] # use this to choose specific sample
    for i, idx in enumerate(indices):
        A0 = geometries[idx]
        
        # Convert Area to Radius: A = pi * r^2  ->  r = sqrt(A / pi)
        r0 = np.sqrt(A0 / np.pi)
        
        ax = axes[i]
        
        # Plot "Top" and "Bottom" walls
        ax.plot(x_grid, r0, color='darkred', linewidth=2, label='Vessel Wall')
        ax.plot(x_grid, -r0, color='darkred', linewidth=2)
        
        # Fill the inside to look like blood
        ax.fill_between(x_grid, -r0, r0, color='mistyrose', alpha=0.5, label='Blood Volume')
        
        # Add a centerline for reference
        ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # Formatting
        ax.set_title(f"Vessel Sample #{idx} (Side View)", fontsize=12)
        ax.set_ylabel("Radius (m)")
        ax.grid(True, alpha=0.3)
        
        # Highlight Stenosis if present (visually)
        # We can detect if min area is significantly smaller than max area
        if np.min(A0) < 0.95 * np.max(A0):
            ax.text(0.05, 0.8 * np.max(r0), "Stenosis Detected", 
                    color='red', fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    axes[-1].set_xlabel("Vessel Length (m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_vessel("vessel_geometries.npy", num_samples=1)