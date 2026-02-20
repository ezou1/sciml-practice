"""
visualize_flows.py — Animate blood flow predictions as a stop-motion GIF.

Draws the vessel side-view at each time step with the interior colored by
flow rate (blue → red).  Vessel walls are drawn as dark red lines.

Usage:
    python visualize_flows.py                        # random vessel, save GIF
    python visualize_flows.py --vessel 42            # specific vessel
    python visualize_flows.py --vessel 42 --show     # display instead of save
    python visualize_flows.py --vessel 42 --fps 30   # higher frame rate
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection


def load_results(path="blood_flow_results.npz"):
    data = np.load(path, allow_pickle=True)
    return {
        "x": data["x"],        # (N_x,) in cm
        "t": data["t"],        # (N_t,) in seconds
        "A": data["A"],        # (num_vessels, N_t, N_x) in cm²
        "q": data["q"],        # (num_vessels, N_t, N_x) in cm³/s
        "A0": data["A0"],      # (num_vessels, m_points) in cm²
    }


def animate_vessel(results, vessel_idx, fps=20, save_path=None, show=False):
    """Create a stop-motion animation of blood flow through one vessel.

    Each frame shows the vessel cross-section (side view) at one time step,
    with the interior colored by local flow rate q.
    """
    x = results["x"]           # spatial grid (cm)
    t = results["t"]           # time grid (s)
    A = results["A"][vessel_idx]  # (N_t, N_x)
    q = results["q"][vessel_idx]  # (N_t, N_x)

    N_t, N_x = A.shape

    # Derive radius from area:  r = sqrt(A / pi)
    R = np.sqrt(A / np.pi)     # (N_t, N_x) in cm

    # Global flow range for consistent colormap across all frames
    q_min, q_max = q.min(), q.max()
    # Ensure we have a reasonable range even if flow is near-zero
    if q_max - q_min < 1e-10:
        q_max = q_min + 1.0
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap = plt.cm.coolwarm  # blue (low/negative) → red (high flow)

    # Resting geometry for reference
    A0 = results["A0"][vessel_idx]
    x0_grid = np.linspace(x[0], x[-1], len(A0))
    r0 = np.sqrt(A0 / np.pi)

    # Detect stenosis
    has_stenosis = np.min(A0) < 0.90 * np.max(A0)

    # ---- Set up figure ----
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.subplots_adjust(bottom=0.18, top=0.88)

    # Colorbar (created once)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Flow rate q (cm³/s)", fontsize=10)

    # Max radius across all time for fixed y-limits
    r_max = R.max() * 1.3

    def draw_frame(frame_idx):
        ax.clear()

        r_t = R[frame_idx]     # radius at this time step, shape (N_x,)
        q_t = q[frame_idx]     # flow at this time step,   shape (N_x,)

        # Draw colored interior: one polygon per x-segment, colored by flow
        verts = []
        colors = []
        for j in range(N_x - 1):
            # Quadrilateral from (x[j], -r) to (x[j+1], -r) to (x[j+1], +r) to (x[j], +r)
            poly = [
                (x[j],   -r_t[j]),
                (x[j+1], -r_t[j+1]),
                (x[j+1],  r_t[j+1]),
                (x[j],    r_t[j]),
            ]
            verts.append(poly)
            # Color by average flow in this segment
            q_avg = 0.5 * (q_t[j] + q_t[j+1])
            colors.append(cmap(norm(q_avg)))

        collection = PolyCollection(verts, facecolors=colors, edgecolors="none")
        ax.add_collection(collection)

        # Draw vessel walls (top and bottom)
        ax.plot(x, r_t, color="darkred", linewidth=2, label="Vessel wall")
        ax.plot(x, -r_t, color="darkred", linewidth=2)

        # Draw resting geometry as dashed reference
        ax.plot(x0_grid,  r0, color="gray", linewidth=1, linestyle="--", alpha=0.5, label="Resting shape")
        ax.plot(x0_grid, -r0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

        # Centerline
        ax.axhline(0, color="black", linestyle="--", alpha=0.2, linewidth=0.5)

        # Labels
        t_ms = t[frame_idx] * 1000  # convert to ms
        ax.set_title(f"Vessel #{vessel_idx}  —  t = {t_ms:.1f} ms  (frame {frame_idx+1}/{N_t})",
                      fontsize=12)
        ax.set_xlabel("Vessel length (cm)")
        ax.set_ylabel("Radius (cm)")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-r_max, r_max)
        ax.grid(True, alpha=0.2)

        # Stenosis label
        if has_stenosis:
            ax.text(0.02, 0.95, "Stenosis present", transform=ax.transAxes,
                    color="red", fontsize=9, fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # ---- Create animation ----
    print(f"Creating animation: {N_t} frames at {fps} fps ({N_t/fps:.1f}s duration)...")
    ani = animation.FuncAnimation(fig, draw_frame, frames=N_t, interval=1000 / fps)

    if save_path:
        print(f"Saving to {save_path}...")
        ani.save(save_path, writer="pillow", fps=fps)
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    elif not save_path:
        # Default: save as GIF
        default_path = f"blood_flow_vessel{vessel_idx}.gif"
        print(f"Saving to {default_path}...")
        ani.save(default_path, writer="pillow", fps=fps)
        print(f"Saved: {default_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize blood flow predictions")
    parser.add_argument("--data", type=str, default="blood_flow_results.npz",
                        help="Path to .npz results file from test.py")
    parser.add_argument("--vessel", type=int, default=None,
                        help="Vessel index (0-999). Random if not specified.")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second for the animation")
    parser.add_argument("--save", type=str, default=None,
                        help="Output file path (e.g. flow.gif or flow.mp4)")
    parser.add_argument("--show", action="store_true",
                        help="Display the animation in a window instead of saving")
    args = parser.parse_args()

    results = load_results(args.data)
    num_vessels = results["A"].shape[0]

    vessel_idx = args.vessel
    if vessel_idx is None:
        vessel_idx = np.random.randint(0, num_vessels)
        print(f"Randomly selected vessel #{vessel_idx}")
    elif vessel_idx < 0 or vessel_idx >= num_vessels:
        print(f"Error: vessel index must be 0–{num_vessels-1}, got {vessel_idx}")
        return

    print(f"Vessel #{vessel_idx}:")
    print(f"  A0 range: [{results['A0'][vessel_idx].min():.4f}, {results['A0'][vessel_idx].max():.4f}] cm²")
    print(f"  A  range: [{results['A'][vessel_idx].min():.4f}, {results['A'][vessel_idx].max():.4f}] cm²")
    print(f"  q  range: [{results['q'][vessel_idx].min():.4f}, {results['q'][vessel_idx].max():.4f}] cm³/s")
    print(f"  Time span: {results['t'][0]*1000:.1f} – {results['t'][-1]*1000:.1f} ms")

    animate_vessel(results, vessel_idx, fps=args.fps, save_path=args.save, show=args.show)


if __name__ == "__main__":
    main()
