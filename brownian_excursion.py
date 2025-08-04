import numpy as np
import matplotlib.pyplot as plt
import random
import os

class DiscreteExcursion:
    """
    Discrete Brownian excursion using nearest-neighbor random walks on Z^d:
    - X-axis: 1D random walk on Z (discrete Brownian motion)
    - Y-axis: Distance from origin of 3D random walk on Z³ (discrete 3D Bessel process)
    
    This gives a purely discrete approximation that converges to Brownian excursions
    in the scaling limit.
    """
    
    def __init__(self):
        # 1D moves for x-coordinate
        self.moves_1d = [-1, 1]
        # 3D moves for computing ||RW³||
        self.moves_3d = [
            (1, 0, 0), (-1, 0, 0),    # ±x
            (0, 1, 0), (0, -1, 0),    # ±y  
            (0, 0, 1), (0, 0, -1)     # ±z
        ]
        
    def random_walk_3d(self, n_steps):
        """
        Generate a 3D random walk on Z³ using nearest-neighbor moves.
        
        Args:
            n_steps: Number of steps to take
            
        Returns:
            List of (x, y, z) positions and distances from origin
        """
        # Start at origin
        x, y, z = 0, 0, 0
        positions = [(x, y, z)]
        distances = [0]  # Distance from origin
        
        for _ in range(n_steps):
            # Choose random 3D move
            dx, dy, dz = random.choice(self.moves_3d)
            x += dx
            y += dy
            z += dz
            
            positions.append((x, y, z))
            # Distance from origin (discrete 3D Bessel process)
            distance = np.sqrt(x*x + y*y + z*z)
            distances.append(distance)
        
        return positions, distances
    
    def random_walk_1d(self, n_steps):
        """
        Generate a 1D random walk on Z using nearest-neighbor moves.
        
        Args:
            n_steps: Number of steps to take
            
        Returns:
            List of positions (discrete Brownian motion)
        """
        position = 0
        positions = [position]
        
        for _ in range(n_steps):
            # Choose random 1D move: -1 or +1
            step = random.choice(self.moves_1d)
            position += step
            positions.append(position)
        
        return positions
    
    def create_discrete_excursion(self, n_steps):
        """
        Create a discrete Brownian excursion in the upper half-plane using
        nearest-neighbor random walks on Z^d.
        
        Args:
            n_steps: Number of steps for both walks
            
        Returns:
            x_values (1D RW), y_values (||3D RW||)
        """
        # X-coordinate: 1D random walk on Z (discrete Brownian motion)
        x_values = self.random_walk_1d(n_steps)
        
        # Y-coordinate: Distance from origin of 3D random walk on Z³
        _, y_values = self.random_walk_3d(n_steps)
        
        return x_values, y_values
    
    def analyze_discrete_excursion(self, x_values, y_values):
        """
        Analyze statistical properties of the discrete excursion.
        """
        n_steps = len(x_values) - 1
        
        # Maximum height
        max_height = max(y_values)
        max_height_step = y_values.index(max_height)
        
        # Total variation
        x_variation = sum(abs(x_values[i+1] - x_values[i]) for i in range(len(x_values)-1))
        y_variation = sum(abs(y_values[i+1] - y_values[i]) for i in range(len(y_values)-1))
        
        # End-to-end distance
        end_to_end = np.sqrt(x_values[-1]**2 + y_values[-1]**2)
        
        # Simple area approximation (sum of y-values)
        area = sum(y_values)
        
        return {
            'n_steps': n_steps,
            'max_height': max_height,
            'max_height_step': max_height_step,
            'x_variation': x_variation,
            'y_variation': y_variation,
            'area': area,
            'end_to_end': end_to_end,
            'final_x': x_values[-1],
            'final_y': y_values[-1]
        }
    

def create_discrete_excursion_visualizations():
    """
    Create visualizations of discrete Brownian excursions using random walks on Z^d.
    """
    
    # Set seeds for reproducibility
    np.random.seed(123)
    random.seed(123)
    
    excursion = DiscreteExcursion()
    
    # Generate multiple excursions with different step counts
    step_counts = [500, 1000, 2000, 3000]
    excursions = []
    
    print("Generating discrete Brownian excursions using Z^d random walks...")
    
    for n_steps in step_counts:
        print(f"Generating excursion with {n_steps} steps...")
        x_vals, y_vals = excursion.create_discrete_excursion(n_steps)
        properties = excursion.analyze_discrete_excursion(x_vals, y_vals)
        excursions.append((n_steps, x_vals, y_vals, properties))
        
        print(f"  Max height: {properties['max_height']:.3f}")
        print(f"  Final position: ({properties['final_x']}, {properties['final_y']:.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (n_steps, x_vals, y_vals, props) in enumerate(excursions):
        ax = axes[i]
        
        # Plot the excursion path
        ax.plot(x_vals, y_vals, color=colors[i], linewidth=1.0, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(x_vals[0], y_vals[0], color='darkgreen', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter(x_vals[-1], y_vals[-1], color='darkred', s=100, marker='s', 
                  label='End', zorder=5)
        
        # Mark maximum height
        max_idx = props['max_height_step']
        ax.scatter(x_vals[max_idx], y_vals[max_idx], color='gold', s=80, marker='^',
                  label=f'Max height: {props["max_height"]:.1f}', zorder=5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=2)
        
        ax.set_title(f'Discrete Excursion (N = {n_steps})\n'
                    f'X: 1D RW on Z, Y: ||3D RW on Z³||\n'
                    f'Area ≈ {props["area"]:.0f}')
        ax.set_xlabel('X (1D Random Walk)')
        ax.set_ylabel('Y (Distance from Origin)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Ensure y-axis starts from 0
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])
    
    plt.tight_layout()
    plt.savefig('images/discrete_excursions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return excursions

def analyze_discrete_excursion_scaling():
    """
    Analyze scaling properties of discrete Brownian excursions.
    Study how various quantities scale with number of steps.
    """
    
    print("Starting discrete excursion scaling analysis...")
    
    np.random.seed(42)
    random.seed(42)
    
    excursion = DiscreteExcursion()
    
    # Test different step counts
    step_counts = [200, 500, 1000, 1500, 2000, 3000, 4000]
    
    results = {
        'step_counts': [],
        'max_heights': [],
        'areas': [],
        'x_variations': [],
        'y_variations': [],
        'end_to_ends': []
    }
    
    for n_steps in step_counts:
        print(f"Analyzing N = {n_steps} steps...")
        
        # Multiple runs for statistics
        max_heights = []
        areas = []
        x_variations = []
        y_variations = []
        end_to_ends = []
        
        n_runs = 15
        for run in range(n_runs):
            x_vals, y_vals = excursion.create_discrete_excursion(n_steps)
            props = excursion.analyze_discrete_excursion(x_vals, y_vals)
            
            max_heights.append(props['max_height'])
            areas.append(props['area'])
            x_variations.append(props['x_variation'])
            y_variations.append(props['y_variation'])
            end_to_ends.append(props['end_to_end'])
        
        # Store averages
        results['step_counts'].append(n_steps)
        results['max_heights'].append(np.mean(max_heights))
        results['areas'].append(np.mean(areas))
        results['x_variations'].append(np.mean(x_variations))
        results['y_variations'].append(np.mean(y_variations))
        results['end_to_ends'].append(np.mean(end_to_ends))
        
        print(f"  Average max height: {np.mean(max_heights):.3f}")
        print(f"  Average area: {np.mean(areas):.1f}")
    
    # Create scaling plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    step_counts = np.array(results['step_counts'])
    max_heights = np.array(results['max_heights'])
    areas = np.array(results['areas'])
    x_variations = np.array(results['x_variations'])
    end_to_ends = np.array(results['end_to_ends'])
    
    # Plot 1: Maximum height scaling
    ax1.loglog(step_counts, max_heights, 'ro-', label='Max Height')
    
    # Fit power law
    log_N = np.log(step_counts)
    log_heights = np.log(max_heights)
    fit_heights = np.polyfit(log_N, log_heights, 1)
    alpha_height = fit_heights[0]
    
    ax1.loglog(step_counts, np.exp(fit_heights[1]) * step_counts**fit_heights[0], 'r--',
              label=f'Fit: H ∝ N^{{{alpha_height:.3f}}}')
    
    # Theoretical scaling: H ~ sqrt(N) for discrete RW
    ax1.loglog(step_counts, np.sqrt(step_counts) * (max_heights[0] / np.sqrt(step_counts[0])), 'k--',
              alpha=0.7, label='N^{1/2} (RW theory)')
    
    ax1.set_xlabel('Number of Steps N')
    ax1.set_ylabel('Maximum Height')
    ax1.set_title(f'Maximum Height Scaling\n(α = {alpha_height:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area scaling
    ax2.loglog(step_counts, areas, 'bo-', label='Excursion Area')
    
    log_areas = np.log(areas)
    fit_areas = np.polyfit(log_N, log_areas, 1)
    alpha_area = fit_areas[0]
    
    ax2.loglog(step_counts, np.exp(fit_areas[1]) * step_counts**fit_areas[0], 'b--',
              label=f'Fit: A ∝ N^{{{alpha_area:.3f}}}')
    
    # Expected scaling: Area ~ N^(3/2) for discrete excursions  
    ax2.loglog(step_counts, step_counts**(3/2) * (areas[0] / step_counts[0]**(3/2)), 'k--',
              alpha=0.7, label='N^{3/2} (theory)')
    
    ax2.set_xlabel('Number of Steps N')
    ax2.set_ylabel('Excursion Area')
    ax2.set_title(f'Area Scaling\n(α = {alpha_area:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variation scaling
    ax3.loglog(step_counts, x_variations, 'go-', label='X Variation (1D RW)')
    
    log_x_var = np.log(x_variations)
    fit_x_var = np.polyfit(log_N, log_x_var, 1)
    alpha_x_var = fit_x_var[0]
    
    ax3.loglog(step_counts, np.exp(fit_x_var[1]) * step_counts**fit_x_var[0], 'g--',
              label=f'Fit: V_X ∝ N^{{{alpha_x_var:.3f}}}')
    
    # For 1D RW: variation ~ N
    ax3.loglog(step_counts, step_counts * (x_variations[0] / step_counts[0]), 'k--',
              alpha=0.7, label='N^1 (RW theory)')
    
    ax3.set_xlabel('Number of Steps N')
    ax3.set_ylabel('Total Variation')
    ax3.set_title(f'X-Variation Scaling\n(α = {alpha_x_var:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: End-to-end distance
    ax4.loglog(step_counts, end_to_ends, 'mo-', label='End-to-End Distance')
    
    log_e2e = np.log(end_to_ends)
    fit_e2e = np.polyfit(log_N, log_e2e, 1)
    alpha_e2e = fit_e2e[0]
    
    ax4.loglog(step_counts, np.exp(fit_e2e[1]) * step_counts**fit_e2e[0], 'm--',
              label=f'Fit: R ∝ N^{{{alpha_e2e:.3f}}}')
    
    # Random walk scaling: R ~ sqrt(N)
    ax4.loglog(step_counts, np.sqrt(step_counts) * (end_to_ends[0] / np.sqrt(step_counts[0])), 'k--',
              alpha=0.7, label='N^{1/2} (RW theory)')
    
    ax4.set_xlabel('Number of Steps N')
    ax4.set_ylabel('End-to-End Distance')
    ax4.set_title(f'End-to-End Scaling\n(α = {alpha_e2e:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/discrete_excursion_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiscrete Excursion Scaling Results:")
    print(f"Maximum height: H ∝ N^{alpha_height:.3f} (expected: N^0.5)")
    print(f"Excursion area: A ∝ N^{alpha_area:.3f} (expected: N^1.5)")
    print(f"X-variation: V ∝ N^{alpha_x_var:.3f} (expected: N^1.0)")
    print(f"End-to-end: R ∝ N^{alpha_e2e:.3f} (expected: N^0.5)")
    
    return results

def create_large_excursion_visualization():
    """
    Create a large discrete Brownian excursion with 10^4 steps 
    and grid-aligned visualization.
    """
    print("Generating large discrete excursion (10^4 steps)...")
    
    np.random.seed(456)
    random.seed(456)
    
    excursion = DiscreteExcursion()
    n_steps = 10000
    
    x_vals, y_vals = excursion.create_discrete_excursion(n_steps)
    properties = excursion.analyze_discrete_excursion(x_vals, y_vals)
    
    print(f"Large excursion properties:")
    print(f"  Max height: {properties['max_height']:.3f}")
    print(f"  Final position: ({properties['final_x']}, {properties['final_y']:.3f})")
    print(f"  Area: {properties['area']:.0f}")
    
    # Create grid-aligned visualization
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Plot the excursion path
    ax.plot(x_vals, y_vals, color='blue', linewidth=0.8, alpha=0.9)
    
    # Mark start and end points
    ax.scatter(x_vals[0], y_vals[0], color='darkgreen', s=150, marker='o', 
              label='Start', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter(x_vals[-1], y_vals[-1], color='darkred', s=150, marker='s', 
              label='End', zorder=5, edgecolors='white', linewidth=2)
    
    # Mark maximum height
    max_idx = properties['max_height_step']
    ax.scatter(x_vals[max_idx], y_vals[max_idx], color='gold', s=120, marker='^',
              label=f'Max height: {properties["max_height"]:.1f}', zorder=5, 
              edgecolors='black', linewidth=1)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=3)
    
    # Get axis ranges
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)
    
    # Set equal aspect ratio and aligned grid
    ax.set_aspect('equal', adjustable='box')
    
    # Create grid with same spacing for both axes
    grid_spacing = max(10, int(max(x_range, y_range) / 20))  # Adaptive grid spacing
    
    # Set tick positions
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = 0, max(y_vals)
    
    # Extend ranges slightly for better visualization
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(-y_padding, y_max + y_padding)
    
    # Set grid with equal spacing
    x_ticks = np.arange(int(x_min/grid_spacing)*grid_spacing, 
                       int(x_max/grid_spacing)*grid_spacing + grid_spacing, 
                       grid_spacing)
    y_ticks = np.arange(0, int(y_max/grid_spacing)*grid_spacing + grid_spacing, 
                       grid_spacing)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    # Style the plot
    ax.set_title(f'Large Discrete Excursion (N = {n_steps:,} steps)\n'
                f'X: 1D RW on Z, Y: ||3D RW on Z³|| (Grid-Aligned)\n'
                f'Area ≈ {properties["area"]:.0f}', fontsize=14, pad=20)
    ax.set_xlabel('X (1D Random Walk)', fontsize=12)
    ax.set_ylabel('Y (Distance from Origin)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.4, linewidth=0.8)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig('images/large_discrete_excursion.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return properties

if __name__ == "__main__":
    print("Discrete Brownian Excursion Simulation using Z^d Random Walks")
    print("X-axis: 1D Random Walk on Z, Y-axis: ||3D Random Walk on Z³||")
    print("=" * 65)
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Generate visualizations
    excursions = create_discrete_excursion_visualizations()
    print(f"Generated {len(excursions)} discrete excursions")
    print("Visualization saved: images/discrete_excursions.png")
    
    # Generate large excursion
    print(f"\n" + "="*65)
    large_props = create_large_excursion_visualization()
    print("Large excursion saved: images/large_discrete_excursion.png")
    
    # Scaling analysis
    print(f"\n" + "="*65)
    results = analyze_discrete_excursion_scaling()
    print("Scaling analysis saved: images/discrete_excursion_scaling.png")
    
    print(f"\nAnalysis complete!")
    print("Discrete excursions using nearest-neighbor random walks on Z^d")
    print("provide clean approximations to continuous Brownian excursions.")