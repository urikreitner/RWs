import random
import matplotlib.pyplot as plt
import numpy as np

def random_walk_1d(steps):
    """
    Generate a 1D random walk.
    
    Args:
        steps: Number of steps to take
        
    Returns:
        List of positions over time
    """
    position = 0
    positions = [position]
    
    for _ in range(steps):
        step = random.choice([-1, 1])
        position += step
        positions.append(position)
    
    return positions

def random_walk_2d(steps):
    """
    Generate a 2D random walk on the integer lattice Z^2.
    Uses 4-connected nearest-neighbor moves only (no diagonals).
    
    Args:
        steps: Number of steps to take
        
    Returns:
        Tuple of (x_positions, y_positions)
    """
    x, y = 0, 0
    x_positions = [x]
    y_positions = [y]
    
    # 4-connected moves: North, South, East, West
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for _ in range(steps):
        dx, dy = random.choice(moves)
        x += dx
        y += dy
        x_positions.append(x)
        y_positions.append(y)
    
    return x_positions, y_positions

def plot_random_walk(positions=None, x_positions=None, y_positions=None):
    """
    Plot the random walk.
    
    Args:
        positions: List of 1D positions (for 1D walk)
        x_positions, y_positions: Lists of 2D positions (for 2D walk)
    """
    plt.figure(figsize=(10, 6))
    
    if positions is not None:
        # 1D random walk
        plt.subplot(1, 2, 1)
        plt.plot(positions)
        plt.title('1D Random Walk')
        plt.xlabel('Step')
        plt.ylabel('Position')
        plt.grid(True)
    
    if x_positions is not None and y_positions is not None:
        # 2D random walk
        plt.subplot(1, 2, 2)
        plt.plot(x_positions, y_positions, 'b-', alpha=0.7)
        plt.plot(x_positions[0], y_positions[0], 'go', markersize=8, label='Start')
        plt.plot(x_positions[-1], y_positions[-1], 'ro', markersize=8, label='End')
        plt.title('2D Random Walk')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    num_steps = 1000
    
    # Generate random walks
    walk_1d = random_walk_1d(num_steps)
    walk_2d_x, walk_2d_y = random_walk_2d(num_steps)
    
    # Print some statistics
    print(f"1D Random Walk after {num_steps} steps:")
    print(f"Final position: {walk_1d[-1]}")
    print(f"Distance from origin: {abs(walk_1d[-1])}")
    print(f"Maximum distance from origin: {max(abs(p) for p in walk_1d)}")
    
    print(f"\n2D Random Walk after {num_steps} steps:")
    final_distance = np.sqrt(walk_2d_x[-1]**2 + walk_2d_y[-1]**2)
    print(f"Final position: ({walk_2d_x[-1]}, {walk_2d_y[-1]})")
    print(f"Distance from origin: {final_distance:.2f}")
    
    # Plot the walks
    plot_random_walk(positions=walk_1d, x_positions=walk_2d_x, y_positions=walk_2d_y)