import random
import matplotlib.pyplot as plt
import numpy as np
import os
from random_walk import random_walk_1d, random_walk_2d

def create_simulation_images():
    """Generate and save multiple random walk simulations as images."""
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Simulation 1: Multiple 1D walks comparison
    plt.figure(figsize=(12, 8))
    
    steps = 1000
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    plt.subplot(2, 2, 1)
    for i in range(5):
        walk = random_walk_1d(steps)
        plt.plot(walk, color=colors[i], alpha=0.7, label=f'Walk {i+1}')
    plt.title('Multiple 1D Random Walks (1000 steps)')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Simulation 2: Single detailed 2D walk
    plt.subplot(2, 2, 2)
    x_pos, y_pos = random_walk_2d(steps)
    plt.plot(x_pos, y_pos, 'b-', alpha=0.7, linewidth=0.8)
    plt.plot(x_pos[0], y_pos[0], 'go', markersize=8, label='Start')
    plt.plot(x_pos[-1], y_pos[-1], 'ro', markersize=8, label='End')
    plt.title('2D Random Walk (1000 steps)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Simulation 3: Distance from origin over time
    plt.subplot(2, 2, 3)
    distances_1d = [abs(pos) for pos in walk]
    distances_2d = [np.sqrt(x**2 + y**2) for x, y in zip(x_pos, y_pos)]
    
    plt.plot(distances_1d, 'b-', label='1D Walk', alpha=0.7)
    plt.plot(distances_2d, 'r-', label='2D Walk', alpha=0.7)
    plt.title('Distance from Origin Over Time')
    plt.xlabel('Step')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Simulation 4: Multiple 2D walks endpoints
    plt.subplot(2, 2, 4)
    endpoints_x = []
    endpoints_y = []
    
    for i in range(50):
        x_walk, y_walk = random_walk_2d(500)
        endpoints_x.append(x_walk[-1])
        endpoints_y.append(y_walk[-1])
    
    plt.scatter(endpoints_x, endpoints_y, alpha=0.6, s=30)
    plt.title('Endpoints of 50 Random Walks (500 steps each)')
    plt.xlabel('Final X Position')
    plt.ylabel('Final Y Position')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/random_walk_simulations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual detailed plots
    
    # Long-term 1D walk
    plt.figure(figsize=(12, 6))
    long_walk = random_walk_1d(5000)
    plt.plot(long_walk, 'b-', alpha=0.8, linewidth=0.8)
    plt.title('Long-term 1D Random Walk (5000 steps)')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True, alpha=0.3)
    plt.savefig('images/long_1d_walk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Complex 2D walk with path highlighting
    plt.figure(figsize=(10, 10))
    x_complex, y_complex = random_walk_2d(2000)
    
    # Create a color gradient for the path
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_complex)))
    
    for i in range(len(x_complex)-1):
        plt.plot([x_complex[i], x_complex[i+1]], [y_complex[i], y_complex[i+1]], 
                color=colors[i], alpha=0.7, linewidth=0.5)
    
    plt.plot(x_complex[0], y_complex[0], 'go', markersize=12, label='Start')
    plt.plot(x_complex[-1], y_complex[-1], 'ro', markersize=12, label='End')
    plt.title('2D Random Walk with Path Gradient (2000 steps)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('images/gradient_2d_walk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical analysis plot
    plt.figure(figsize=(12, 8))
    
    # Generate data for statistical analysis
    walk_lengths = [100, 500, 1000, 2000, 5000]
    final_distances_1d = []
    final_distances_2d = []
    max_distances_1d = []
    max_distances_2d = []
    
    for length in walk_lengths:
        # Multiple runs for each length
        distances_1d_runs = []
        distances_2d_runs = []
        max_dist_1d_runs = []
        max_dist_2d_runs = []
        
        for _ in range(20):  # 20 runs per length
            walk_1d = random_walk_1d(length)
            walk_2d_x, walk_2d_y = random_walk_2d(length)
            
            distances_1d_runs.append(abs(walk_1d[-1]))
            distances_2d_runs.append(np.sqrt(walk_2d_x[-1]**2 + walk_2d_y[-1]**2))
            max_dist_1d_runs.append(max(abs(pos) for pos in walk_1d))
            max_dist_2d_runs.append(max(np.sqrt(x**2 + y**2) for x, y in zip(walk_2d_x, walk_2d_y)))
        
        final_distances_1d.append(np.mean(distances_1d_runs))
        final_distances_2d.append(np.mean(distances_2d_runs))
        max_distances_1d.append(np.mean(max_dist_1d_runs))
        max_distances_2d.append(np.mean(max_dist_2d_runs))
    
    plt.subplot(2, 2, 1)
    plt.plot(walk_lengths, final_distances_1d, 'bo-', label='1D Walks')
    plt.plot(walk_lengths, final_distances_2d, 'ro-', label='2D Walks')
    plt.title('Average Final Distance vs Walk Length')
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Distance from Origin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(walk_lengths, max_distances_1d, 'bo-', label='1D Walks')
    plt.plot(walk_lengths, max_distances_2d, 'ro-', label='2D Walks')
    plt.title('Average Maximum Distance vs Walk Length')
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Maximum Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Theoretical vs actual for square root scaling
    plt.subplot(2, 2, 3)
    theoretical = [np.sqrt(length) for length in walk_lengths]
    plt.plot(walk_lengths, theoretical, 'k--', label='√n theoretical', linewidth=2)
    plt.plot(walk_lengths, final_distances_1d, 'bo-', label='1D actual')
    plt.plot(walk_lengths, final_distances_2d, 'ro-', label='2D actual')
    plt.title('Theoretical vs Actual Scaling')
    plt.xlabel('Number of Steps')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution of final positions
    plt.subplot(2, 2, 4)
    final_positions_1d = []
    final_positions_2d = []
    
    for _ in range(1000):
        walk_1d = random_walk_1d(1000)
        walk_2d_x, walk_2d_y = random_walk_2d(1000)
        final_positions_1d.append(walk_1d[-1])
        final_positions_2d.append(np.sqrt(walk_2d_x[-1]**2 + walk_2d_y[-1]**2))
    
    plt.hist(final_positions_1d, bins=50, alpha=0.7, label='1D Final Positions', density=True)
    plt.hist(final_positions_2d, bins=50, alpha=0.7, label='2D Final Distances', density=True)
    plt.title('Distribution of Final Positions (1000 runs)')
    plt.xlabel('Position/Distance')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated simulation images:")
    print("- images/random_walk_simulations.png")
    print("- images/long_1d_walk.png") 
    print("- images/gradient_2d_walk.png")
    print("- images/statistical_analysis.png")

if __name__ == "__main__":
    create_simulation_images()