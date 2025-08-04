import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy import stats

class BrownianExcursion:
    """
    Brownian excursion in the upper half-plane:
    - X-axis: Standard Brownian motion
    - Y-axis: 3D Bessel process (distance from origin of 3D Brownian motion)
    
    The 3D Bessel process stays positive and has the SDE:
    dR_t = dW_t + (1/R_t) dt  (for dimension d=3)
    
    This creates beautiful excursion paths that stay in the upper half-plane.
    """
    
    def __init__(self, dt=0.001):
        self.dt = dt
        
    def simulate_3d_bessel_process(self, T, R0=1.0):
        """
        Simulate a 3D Bessel process using Euler-Maruyama scheme.
        
        The 3D Bessel process R_t satisfies:
        dR_t = dW_t + (1/R_t) dt
        
        where W_t is a Brownian motion and the drift term 1/R_t comes from
        the dimension d=3 (general formula: (d-1)/(2R_t)).
        
        Args:
            T: Final time
            R0: Initial value (must be > 0)
            
        Returns:
            times, values of the Bessel process
        """
        N = int(T / self.dt)
        times = np.linspace(0, T, N + 1)
        R = np.zeros(N + 1)
        R[0] = R0
        
        sqrt_dt = np.sqrt(self.dt)
        
        for i in range(N):
            # Brownian increment
            dW = np.random.normal(0, sqrt_dt)
            
            # Euler-Maruyama step for 3D Bessel process
            # dR = dW + (1/R) dt
            drift = 1.0 / R[i] * self.dt if R[i] > 0 else 0
            R[i + 1] = R[i] + dW + drift
            
            # Ensure positivity (reflection at 0)
            if R[i + 1] <= 0:
                R[i + 1] = abs(R[i + 1]) + 1e-6
        
        return times, R
    
    def simulate_3d_bessel_exact(self, T, R0=1.0):
        """
        Alternative simulation using the exact representation:
        3D Bessel process = ||B_t^3|| where B_t^3 is 3D Brownian motion.
        """
        N = int(T / self.dt)
        times = np.linspace(0, T, N + 1)
        
        # Simulate 3D Brownian motion
        sqrt_dt = np.sqrt(self.dt)
        dW = np.random.normal(0, sqrt_dt, (N, 3))
        
        # Cumulative sum to get Brownian motion paths
        B = np.zeros((N + 1, 3))
        B[1:] = np.cumsum(dW, axis=0)
        
        # Add initial condition to match R0
        # Scale initial position so ||B_0|| = R0
        B[0] = np.array([R0, 0, 0])
        
        # 3D Bessel process is the Euclidean norm
        R = np.linalg.norm(B, axis=1)
        
        return times, R
    
    def simulate_brownian_motion(self, T):
        """
        Simulate standard Brownian motion.
        
        Args:
            T: Final time
            
        Returns:
            times, values of Brownian motion
        """
        N = int(T / self.dt)
        times = np.linspace(0, T, N + 1)
        
        sqrt_dt = np.sqrt(self.dt)
        dW = np.random.normal(0, sqrt_dt, N)
        
        W = np.zeros(N + 1)
        W[1:] = np.cumsum(dW)
        
        return times, W
    
    def create_excursion(self, T=1.0, method='exact'):
        """
        Create a Brownian excursion in the upper half-plane.
        
        Args:
            T: Duration of the excursion
            method: 'exact' or 'sde' for Bessel process simulation
            
        Returns:
            times, x_values (BM), y_values (3D Bessel)
        """
        # Simulate x-coordinate as Brownian motion
        times, x_values = self.simulate_brownian_motion(T)
        
        # Simulate y-coordinate as 3D Bessel process
        if method == 'exact':
            _, y_values = self.simulate_3d_bessel_exact(T, R0=0.5)
        else:
            _, y_values = self.simulate_3d_bessel_process(T, R0=0.5)
        
        return times, x_values, y_values
    
    def analyze_excursion_properties(self, times, x_values, y_values):
        """
        Analyze statistical properties of the excursion.
        """
        T = times[-1]
        N = len(times)
        
        # Maximum height
        max_height = np.max(y_values)
        max_height_time = times[np.argmax(y_values)]
        
        # Total variation
        x_variation = np.sum(np.abs(np.diff(x_values)))
        y_variation = np.sum(np.abs(np.diff(y_values)))
        
        # Quadratic variation (should be ~ T for BM, different for Bessel)
        x_quad_var = np.sum(np.diff(x_values)**2)
        y_quad_var = np.sum(np.diff(y_values)**2)
        
        # Area under the curve
        area = np.trapezoid(y_values, x_values)
        
        # End-to-end distance
        end_to_end = np.sqrt(x_values[-1]**2 + y_values[-1]**2)
        
        return {
            'duration': T,
            'max_height': max_height,
            'max_height_time': max_height_time,
            'x_variation': x_variation,
            'y_variation': y_variation,
            'x_quad_variation': x_quad_var,
            'y_quad_variation': y_quad_var,
            'area': area,
            'end_to_end': end_to_end,
            'final_x': x_values[-1],
            'final_y': y_values[-1]
        }

def create_excursion_visualizations():
    """
    Create visualizations of Brownian excursions in upper half-plane.
    """
    
    # Set seeds for reproducibility
    np.random.seed(123)
    random.seed(123)
    
    excursion = BrownianExcursion(dt=0.001)
    
    # Generate multiple excursions
    durations = [0.5, 1.0, 2.0, 3.0]
    excursions = []
    
    print("Generating Brownian excursions in upper half-plane...")
    
    for T in durations:
        print(f"Generating excursion of duration {T}...")
        times, x_vals, y_vals = excursion.create_excursion(T, method='exact')
        properties = excursion.analyze_excursion_properties(times, x_vals, y_vals)
        excursions.append((T, times, x_vals, y_vals, properties))
        
        print(f"  Max height: {properties['max_height']:.3f}")
        print(f"  Final position: ({properties['final_x']:.3f}, {properties['final_y']:.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (T, times, x_vals, y_vals, props) in enumerate(excursions):
        ax = axes[i]
        
        # Plot the excursion path
        ax.plot(x_vals, y_vals, color=colors[i], linewidth=1.5, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(x_vals[0], y_vals[0], color='darkgreen', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter(x_vals[-1], y_vals[-1], color='darkred', s=100, marker='s', 
                  label='End', zorder=5)
        
        # Mark maximum height
        max_idx = np.argmax(y_vals)
        ax.scatter(x_vals[max_idx], y_vals[max_idx], color='gold', s=80, marker='^',
                  label=f'Max height: {props["max_height"]:.2f}', zorder=5)
        
        # Add horizontal line at y=0
        xlim = ax.get_xlim()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=2)
        ax.fill_between(x_vals, 0, y_vals, alpha=0.2, color=colors[i])
        
        ax.set_title(f'Brownian Excursion (T = {T})\n'
                    f'X: BM, Y: 3D Bessel Process\n'
                    f'Area = {props["area"]:.2f}')
        ax.set_xlabel('X (Brownian Motion)')
        ax.set_ylabel('Y (3D Bessel Process)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ensure y-axis starts from 0
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
    
    plt.tight_layout()
    plt.savefig('images/brownian_excursions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return excursions

def analyze_excursion_scaling():
    """
    Analyze scaling properties of Brownian excursions.
    Study how various quantities scale with excursion duration.
    """
    
    print("Starting Brownian excursion scaling analysis...")
    
    np.random.seed(42)
    random.seed(42)
    
    excursion = BrownianExcursion(dt=0.002)  # Slightly coarser for speed
    
    # Test different durations
    durations = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    results = {
        'durations': [],
        'max_heights': [],
        'areas': [],
        'x_variations': [],
        'y_variations': [],
        'end_to_ends': []
    }
    
    for T in durations:
        print(f"Analyzing duration T = {T}...")
        
        # Multiple runs for statistics
        max_heights = []
        areas = []
        x_variations = []
        y_variations = []
        end_to_ends = []
        
        n_runs = 20
        for run in range(n_runs):
            times, x_vals, y_vals = excursion.create_excursion(T, method='exact')
            props = excursion.analyze_excursion_properties(times, x_vals, y_vals)
            
            max_heights.append(props['max_height'])
            areas.append(abs(props['area']))  # Take absolute value
            x_variations.append(props['x_variation'])
            y_variations.append(props['y_variation'])
            end_to_ends.append(props['end_to_end'])
        
        # Store averages
        results['durations'].append(T)
        results['max_heights'].append(np.mean(max_heights))
        results['areas'].append(np.mean(areas))
        results['x_variations'].append(np.mean(x_variations))
        results['y_variations'].append(np.mean(y_variations))
        results['end_to_ends'].append(np.mean(end_to_ends))
        
        print(f"  Average max height: {np.mean(max_heights):.3f}")
        print(f"  Average area: {np.mean(areas):.3f}")
    
    # Create scaling plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    durations = np.array(results['durations'])
    max_heights = np.array(results['max_heights'])
    areas = np.array(results['areas'])
    x_variations = np.array(results['x_variations'])
    end_to_ends = np.array(results['end_to_ends'])
    
    # Plot 1: Maximum height scaling
    ax1.loglog(durations, max_heights, 'ro-', label='Max Height')
    
    # Fit power law
    log_T = np.log(durations)
    log_heights = np.log(max_heights)
    fit_heights = np.polyfit(log_T, log_heights, 1)
    alpha_height = fit_heights[0]
    
    ax1.loglog(durations, np.exp(fit_heights[1]) * durations**fit_heights[0], 'r--',
              label=f'Fit: H ∝ T^{{{alpha_height:.3f}}}')
    
    # Theoretical scaling for Brownian motion: H ~ sqrt(T)
    ax1.loglog(durations, np.sqrt(durations) * (max_heights[0] / np.sqrt(durations[0])), 'k--',
              alpha=0.7, label='T^{1/2} (BM theory)')
    
    ax1.set_xlabel('Duration T')
    ax1.set_ylabel('Maximum Height')
    ax1.set_title(f'Maximum Height Scaling\n(α = {alpha_height:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area scaling
    ax2.loglog(durations, areas, 'bo-', label='Excursion Area')
    
    log_areas = np.log(areas)
    fit_areas = np.polyfit(log_T, log_areas, 1)
    alpha_area = fit_areas[0]
    
    ax2.loglog(durations, np.exp(fit_areas[1]) * durations**fit_areas[0], 'b--',
              label=f'Fit: A ∝ T^{{{alpha_area:.3f}}}')
    
    # Expected scaling: Area ~ T^(3/2) for Brownian excursions
    ax2.loglog(durations, durations**(3/2) * (areas[0] / durations[0]**(3/2)), 'k--',
              alpha=0.7, label='T^{3/2} (theory)')
    
    ax2.set_xlabel('Duration T')
    ax2.set_ylabel('Excursion Area')
    ax2.set_title(f'Area Scaling\n(α = {alpha_area:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variation scaling
    ax3.loglog(durations, x_variations, 'go-', label='X Variation (BM)')
    
    log_x_var = np.log(x_variations)
    fit_x_var = np.polyfit(log_T, log_x_var, 1)
    alpha_x_var = fit_x_var[0]
    
    ax3.loglog(durations, np.exp(fit_x_var[1]) * durations**fit_x_var[0], 'g--',
              label=f'Fit: V_X ∝ T^{{{alpha_x_var:.3f}}}')
    
    ax3.set_xlabel('Duration T')
    ax3.set_ylabel('Total Variation')
    ax3.set_title(f'X-Variation Scaling\n(α = {alpha_x_var:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: End-to-end distance
    ax4.loglog(durations, end_to_ends, 'mo-', label='End-to-End Distance')
    
    log_e2e = np.log(end_to_ends)
    fit_e2e = np.polyfit(log_T, log_e2e, 1)
    alpha_e2e = fit_e2e[0]
    
    ax4.loglog(durations, np.exp(fit_e2e[1]) * durations**fit_e2e[0], 'm--',
              label=f'Fit: R ∝ T^{{{alpha_e2e:.3f}}}')
    
    # Brownian motion scaling: R ~ sqrt(T)
    ax4.loglog(durations, np.sqrt(durations) * (end_to_ends[0] / np.sqrt(durations[0])), 'k--',
              alpha=0.7, label='T^{1/2} (BM theory)')
    
    ax4.set_xlabel('Duration T')
    ax4.set_ylabel('End-to-End Distance')
    ax4.set_title(f'End-to-End Scaling\n(α = {alpha_e2e:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/excursion_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nScaling Results:")
    print(f"Maximum height: H ∝ T^{alpha_height:.3f} (expected: T^0.5)")
    print(f"Excursion area: A ∝ T^{alpha_area:.3f} (expected: T^1.5)")
    print(f"X-variation: V ∝ T^{alpha_x_var:.3f}")
    print(f"End-to-end: R ∝ T^{alpha_e2e:.3f} (expected: T^0.5)")
    
    return results

if __name__ == "__main__":
    print("Brownian Excursion Simulation in Upper Half-Plane")
    print("X-axis: Brownian Motion, Y-axis: 3D Bessel Process")
    print("=" * 60)
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Generate visualizations
    excursions = create_excursion_visualizations()
    print(f"Generated {len(excursions)} Brownian excursions")
    print("Visualization saved: images/brownian_excursions.png")
    
    # Scaling analysis
    print(f"\n" + "="*60)
    results = analyze_excursion_scaling()
    print("Scaling analysis saved: images/excursion_scaling_analysis.png")
    
    print(f"\nAnalysis complete!")
    print("Brownian excursions in upper half-plane show rich scaling behavior")
    print("combining properties of Brownian motion and 3D Bessel processes.")