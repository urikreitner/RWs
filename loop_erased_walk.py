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
    Get the actual outer boundary of the random walk by finding the perimeter
    of the visited points (not just the convex hull).
    
    Args:
        x_positions: List of x coordinates
        y_positions: List of y coordinates
        
    Returns:
        Tuple of (boundary_x, boundary_y) coordinates of outer boundary
    """
    if len(x_positions) < 3:
        return x_positions, y_positions
    
    # Create a set of all visited points
    points = set(zip(x_positions, y_positions))
    
    # Find boundary points - points that have at least one neighbor not in the set
    boundary_points = []
    
    for x, y in points:
        # Check 8-connected neighbors
        neighbors = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),             (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        
        # If any neighbor is not visited, this is a boundary point
        is_boundary = any(neighbor not in points for neighbor in neighbors)
        
        if is_boundary:
            boundary_points.append((x, y))
    
    if not boundary_points:
        return x_positions, y_positions
    
    # Sort boundary points to create a connected path
    # Start from leftmost point, then go counterclockwise
    boundary_points.sort(key=lambda p: (p[0], p[1]))
    start_point = boundary_points[0]
    
    # Order points by angle from start point to create a closed boundary
    def angle_from_start(point):
        dx = point[0] - start_point[0]
        dy = point[1] - start_point[1]
        return np.arctan2(dy, dx)
    
    boundary_points.sort(key=angle_from_start)
    
    # Close the boundary
    if boundary_points and boundary_points[0] != boundary_points[-1]:
        boundary_points.append(boundary_points[0])
    
    if boundary_points:
        boundary_x, boundary_y = zip(*boundary_points)
        return list(boundary_x), list(boundary_y)
    else:
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