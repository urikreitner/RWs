import random
import matplotlib.pyplot as plt
import numpy as np
from random_walk import random_walk_2d

def loop_erased_random_walk(x_positions, y_positions):
    """
    Perform loop erasure on a random walk path.
    
    Args:
        x_positions: List of x coordinates of the walk
        y_positions: List of y coordinates of the walk
        
    Returns:
        Tuple of (erased_x, erased_y) with loops removed
    """
    if len(x_positions) != len(y_positions):
        raise ValueError("x_positions and y_positions must have the same length")
    
    # Create list of (x, y) positions
    positions = list(zip(x_positions, y_positions))
    
    # Keep track of visited positions and their first occurrence
    visited = {}
    erased_path = []
    
    for i, pos in enumerate(positions):
        if pos in visited:
            # Found a loop - erase everything between first occurrence and now
            first_occurrence = visited[pos]
            # Keep path up to first occurrence, then continue from current position
            erased_path = erased_path[:first_occurrence + 1]
            # Update visited dict to only include positions before the loop
            visited = {p: j for j, p in enumerate(erased_path)}
        else:
            # New position - add to path
            visited[pos] = len(erased_path)
            erased_path.append(pos)
    
    # Separate x and y coordinates
    if erased_path:
        erased_x, erased_y = zip(*erased_path)
        return list(erased_x), list(erased_y)
    else:
        return [], []

def get_outer_boundary(x_positions, y_positions):
    """
    Get the actual outer boundary using the facial walk algorithm for planar graphs.
    This is a linear-time O(N) algorithm that traces the boundary of the unbounded face.
    
    Based on the right-hand rule facial walk algorithm used for extracting
    outer boundaries of random walks in SLE research.
    
    Args:
        x_positions: List of x coordinates of the walk
        y_positions: List of y coordinates of the walk
        
    Returns:
        Tuple of (boundary_x, boundary_y) coordinates of outer boundary
    """
    import math
    from collections import defaultdict
    
    if len(x_positions) < 3:
        return x_positions, y_positions
    
    # Convert walk to list of vertices
    walk = list(zip(x_positions, y_positions))
    
    # 0. Build adjacency (each undirected edge stored once)
    adj = defaultdict(set)
    for a, b in zip(walk, walk[1:]):
        if a == b:  # ignore null steps if any
            continue
        adj[a].add(b)
        adj[b].add(a)
    
    if not adj:
        return x_positions, y_positions
    
    # 1. Pick deterministic starting vertex v0 (lowest y, then x)
    v0 = min(adj.keys(), key=lambda p: (p[1], p[0]))
    incoming = (0, -1)  # start "pointing south"
    
    def angle(ccw_from, to):
        """Signed angle in [0, 2π) between vectors 'ccw_from'→'to' CCW-wise"""
        dx1, dy1 = ccw_from
        dx2, dy2 = to
        phi1 = math.atan2(dy1, dx1)
        phi2 = math.atan2(dy2, dx2)
        a = phi2 - phi1
        return a if a >= 0 else a + 2*math.pi
    
    boundary = [v0]
    v = v0
    max_iterations = len(adj) * 4  # Safety limit
    iterations = 0
    
    while iterations < max_iterations:
        # 2. Among neighbors choose the one that gives *largest* clockwise turn
        candidates = adj[v]
        if not candidates:
            break
            
        best = max(candidates, key=lambda w: angle(incoming, (w[0]-v[0], w[1]-v[1])))
        incoming = (v[0]-best[0], v[1]-best[1])  # new incoming = -(chosen step)
        v = best
        
        if v == v0:
            break
        boundary.append(v)
        iterations += 1
    
    if boundary:
        boundary_x, boundary_y = zip(*boundary)
        return list(boundary_x), list(boundary_y)
    
    return x_positions, y_positions

def create_loop_erased_visualization():
    """Create visualization showing original walk, convex hull, and loop-erased path."""
    
    # Set seed for reproducible results
    random.seed(123)
    np.random.seed(123)
    
    # Generate a 2D random walk with many more steps
    steps = 10000
    x_original, y_original = random_walk_2d(steps)
    
    # Perform loop erasure
    x_erased, y_erased = loop_erased_random_walk(x_original, y_original)
    
    # Get the actual outer boundary of visited points
    boundary_x, boundary_y = get_outer_boundary(x_original, y_original)
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Plot original walk path (light gray, thin line)
    plt.plot(x_original, y_original, color='lightgray', linewidth=0.5, alpha=0.7, 
             label=f'Original Walk ({len(x_original)} steps)')
    
    # Plot outer boundary (blue, thick line)
    plt.plot(boundary_x, boundary_y, color='blue', linewidth=2, alpha=0.8, 
             label='Outer Boundary')
    
    # Plot loop-erased path (red, medium line)
    plt.plot(x_erased, y_erased, color='red', linewidth=2, alpha=0.9,
             label=f'Loop-Erased Path ({len(x_erased)} steps)')
    
    # Mark start and end points
    plt.plot(x_original[0], y_original[0], 'go', markersize=10, label='Start', zorder=5)
    plt.plot(x_original[-1], y_original[-1], 'ko', markersize=8, label='End (Original)', zorder=5)
    
    if x_erased and y_erased:
        plt.plot(x_erased[-1], y_erased[-1], 'ro', markersize=8, label='End (Loop-Erased)', zorder=5)
    
    # Formatting
    plt.title('2D Random Walk: Original Path, Outer Boundary, and Loop Erasure', fontsize=14)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add statistics text box
    original_length = len(x_original)
    erased_length = len(x_erased)
    reduction_percent = (1 - erased_length / original_length) * 100 if original_length > 0 else 0
    
    stats_text = f"""Statistics:
Original steps: {original_length}
Loop-erased steps: {erased_length}
Reduction: {reduction_percent:.1f}%
Boundary points: {len(boundary_x)}"""
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/loop_erased_walk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated loop-erased walk visualization:")
    print(f"- Original walk: {original_length} steps")
    print(f"- Loop-erased walk: {erased_length} steps")
    print(f"- Reduction: {reduction_percent:.1f}%")
    print(f"- Image saved as: images/loop_erased_walk.png")
    
    return x_original, y_original, x_erased, y_erased, boundary_x, boundary_y

if __name__ == "__main__":
    create_loop_erased_visualization()