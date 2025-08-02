import numpy as np
import matplotlib.pyplot as plt
import random
import os
from random_walk import random_walk_2d
from loop_erased_walk import loop_erased_random_walk, get_outer_boundary

def estimate_hausdorff_dimension():
    """
    Estimate the Hausdorff dimension of:
    1. Random walk path
    2. Loop-erased random walk 
    3. Outer boundary (frontier)
    
    Uses box-counting method with multiple walk lengths and multiple runs.
    """
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Walk lengths to test (powers of 2 for clean scaling)
    walk_lengths = [2**i for i in range(8, 16)]  # 256 to 65536 steps
    print(f"Testing walk lengths: {walk_lengths}")
    
    # Number of runs for each length
    runs_per_length = 20
    
    # Results storage
    results = {
        'walk_lengths': [],
        'rw_perimeters': [],
        'lerw_lengths': [], 
        'boundary_lengths': [],
        'rw_areas': [],
        'boundary_areas': []
    }
    
    print("Running simulations...")
    
    for i, n_steps in enumerate(walk_lengths):
        print(f"Processing length {n_steps} ({i+1}/{len(walk_lengths)})...")
        
        run_rw_perimeters = []
        run_lerw_lengths = []
        run_boundary_lengths = []
        run_rw_areas = []
        run_boundary_areas = []
        
        for run in range(runs_per_length):
            # Generate random walk
            x_pos, y_pos = random_walk_2d(n_steps)
            
            # Calculate random walk perimeter (unique positions)
            unique_positions = set(zip(x_pos, y_pos))
            rw_perimeter = len(unique_positions)
            
            # Calculate area (bounding box area)
            min_x, max_x = min(x_pos), max(x_pos)
            min_y, max_y = min(y_pos), max(y_pos)
            rw_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            
            # Generate loop-erased walk
            lerw_x, lerw_y = loop_erased_random_walk(x_pos, y_pos)
            lerw_length = len(lerw_x)
            
            # Generate outer boundary
            try:
                boundary_x, boundary_y = get_outer_boundary(x_pos, y_pos)
                boundary_length = len(boundary_x)
                
                # Calculate boundary area (area enclosed by boundary)
                if len(boundary_x) >= 3:
                    # Use shoelace formula for polygon area
                    boundary_area = 0.5 * abs(sum(
                        boundary_x[i] * boundary_y[i+1] - boundary_x[i+1] * boundary_y[i]
                        for i in range(len(boundary_x)-1)
                    ))
                else:
                    boundary_area = 0
                    
            except:
                boundary_length = 0
                boundary_area = 0
            
            run_rw_perimeters.append(rw_perimeter)
            run_lerw_lengths.append(lerw_length)
            run_boundary_lengths.append(boundary_length)
            run_rw_areas.append(rw_area)
            run_boundary_areas.append(boundary_area)
        
        # Store averages
        results['walk_lengths'].append(n_steps)
        results['rw_perimeters'].append(np.mean(run_rw_perimeters))
        results['lerw_lengths'].append(np.mean(run_lerw_lengths))
        results['boundary_lengths'].append(np.mean(run_boundary_lengths))
        results['rw_areas'].append(np.mean(run_rw_areas))
        results['boundary_areas'].append(np.mean(run_boundary_areas))
    
    return results

def plot_scaling_analysis(results):
    """
    Create scaling plots to estimate Hausdorff dimensions.
    """
    
    walk_lengths = np.array(results['walk_lengths'])
    rw_perimeters = np.array(results['rw_perimeters'])
    lerw_lengths = np.array(results['lerw_lengths'])
    boundary_lengths = np.array(results['boundary_lengths'])
    rw_areas = np.array(results['rw_areas'])
    boundary_areas = np.array(results['boundary_areas'])
    
    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Walk length vs various measures
    ax1 = axes[0, 0]
    ax1.loglog(walk_lengths, rw_perimeters, 'bo-', label='RW Unique Sites', alpha=0.7)
    ax1.loglog(walk_lengths, lerw_lengths, 'ro-', label='LERW Length', alpha=0.7)
    ax1.loglog(walk_lengths, boundary_lengths, 'go-', label='Boundary Length', alpha=0.7)
    
    # Add theoretical scaling lines
    ax1.loglog(walk_lengths, walk_lengths**0.5, 'k--', alpha=0.5, label='N^{1/2} (RW sites)')
    ax1.loglog(walk_lengths, walk_lengths**0.625, 'r--', alpha=0.5, label='N^{5/8} (LERW)')
    ax1.loglog(walk_lengths, walk_lengths**(4/3), 'g--', alpha=0.5, label='N^{4/3} (Boundary)')
    
    ax1.set_xlabel('Walk Length N')
    ax1.set_ylabel('Length/Count')
    ax1.set_title('Scaling of Walk Measures vs Walk Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area scaling
    ax2 = axes[0, 1]
    ax2.loglog(walk_lengths, rw_areas, 'bo-', label='RW Bounding Box Area', alpha=0.7)
    ax2.loglog(walk_lengths, boundary_areas, 'go-', label='Boundary Enclosed Area', alpha=0.7)
    
    # Theoretical area scaling
    ax2.loglog(walk_lengths, walk_lengths, 'k--', alpha=0.5, label='N^1 (RW area)')
    ax2.loglog(walk_lengths, walk_lengths**(3/2), 'g--', alpha=0.5, label='N^{3/2} (Boundary area)')
    
    ax2.set_xlabel('Walk Length N')
    ax2.set_ylabel('Area')
    ax2.set_title('Area Scaling vs Walk Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Dimension estimation from fits
    ax3 = axes[1, 0]
    
    # Fit power laws and extract exponents
    log_N = np.log(walk_lengths)
    
    # RW sites scaling
    log_rw_sites = np.log(rw_perimeters)
    rw_fit = np.polyfit(log_N, log_rw_sites, 1)
    rw_dim = rw_fit[0]
    
    # LERW scaling  
    log_lerw = np.log(lerw_lengths)
    lerw_fit = np.polyfit(log_N, log_lerw, 1)
    lerw_dim = lerw_fit[0]
    
    # Boundary scaling
    log_boundary = np.log(boundary_lengths)
    boundary_fit = np.polyfit(log_N, log_boundary, 1)
    boundary_dim = boundary_fit[0]
    
    # Plot fitted lines
    ax3.loglog(walk_lengths, np.exp(rw_fit[1]) * walk_lengths**rw_fit[0], 'b-', 
               label=f'RW Sites: α = {rw_dim:.3f}')
    ax3.loglog(walk_lengths, np.exp(lerw_fit[1]) * walk_lengths**lerw_fit[0], 'r-',
               label=f'LERW: α = {lerw_dim:.3f}')
    ax3.loglog(walk_lengths, np.exp(boundary_fit[1]) * walk_lengths**boundary_fit[0], 'g-',
               label=f'Boundary: α = {boundary_dim:.3f}')
    
    # Plot data points
    ax3.loglog(walk_lengths, rw_perimeters, 'bo', alpha=0.6)
    ax3.loglog(walk_lengths, lerw_lengths, 'ro', alpha=0.6)
    ax3.loglog(walk_lengths, boundary_lengths, 'go', alpha=0.6)
    
    ax3.set_xlabel('Walk Length N')
    ax3.set_ylabel('Length/Count')
    ax3.set_title('Power Law Fits and Scaling Exponents')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Boundary length vs boundary area (different scaling)
    ax4 = axes[1, 1]
    valid_areas = boundary_areas > 0
    if np.any(valid_areas):
        areas_valid = boundary_areas[valid_areas]
        lengths_valid = boundary_lengths[valid_areas]
        
        ax4.loglog(areas_valid, lengths_valid, 'go-', alpha=0.7, label='Boundary Data')
        
        # Theoretical scaling: Length ~ Area^(D/2) where D is Hausdorff dimension
        # For D = 4/3: Length ~ Area^(2/3)
        if len(areas_valid) > 0:
            ax4.loglog(areas_valid, areas_valid**(2/3) * (lengths_valid[0] / areas_valid[0]**(2/3)), 
                      'k--', alpha=0.5, label='Area^{2/3} (D=4/3)')
        
        ax4.set_xlabel('Boundary Area')
        ax4.set_ylabel('Boundary Length')
        ax4.set_title('Boundary Length vs Area Scaling')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/hausdorff_dimension_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print results
    print(f"\nScaling Exponent Results:")
    print(f"Random Walk Sites: α = {rw_dim:.3f} (theoretical: 0.5)")
    print(f"Loop-Erased Walk: α = {lerw_dim:.3f} (theoretical: 0.625)")
    print(f"Boundary Length: α = {boundary_dim:.3f} (theoretical: 1.333)")
    print(f"\nHausdorff Dimension Estimates:")
    print(f"Random Walk Sites: D ≈ {2 * rw_dim:.3f} (theoretical: 1.0)")
    print(f"Loop-Erased Walk: D ≈ {1 / lerw_dim:.3f} (theoretical: 1.6)")
    print(f"Boundary: D ≈ {boundary_dim:.3f} (theoretical: 4/3 ≈ 1.333)")
    
    return {
        'rw_exponent': rw_dim,
        'lerw_exponent': lerw_dim, 
        'boundary_exponent': boundary_dim
    }

if __name__ == "__main__":
    print("Starting Hausdorff dimension estimation for nearest-neighbor random walks on Z²...")
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Run simulations
    results = estimate_hausdorff_dimension()
    
    # Analyze and plot
    exponents = plot_scaling_analysis(results)
    
    print(f"\nResults saved to images/hausdorff_dimension_analysis.png")
    print("Analysis complete!")