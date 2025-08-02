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

def get_convex_hull(x_positions, y_positions):
    """
    Get the convex hull (outer boundary) of the random walk points.
    
    Args:
        x_positions: List of x coordinates
        y_positions: List of y coordinates
        
    Returns:
        Tuple of (hull_x, hull_y) coordinates of convex hull
    """
    from scipy.spatial import ConvexHull
    
    if len(x_positions) < 3:
        return x_positions, y_positions
    
    points = np.column_stack((x_positions, y_positions))
    
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        # Close the hull by adding the first point at the end
        hull_points = np.vstack([hull_points, hull_points[0]])
        return hull_points[:, 0], hull_points[:, 1]
    except:
        # If convex hull fails (e.g., all points collinear), return original points
        return x_positions, y_positions

def create_loop_erased_visualization():
    """Create visualization showing original walk, convex hull, and loop-erased path."""
    
    # Set seed for reproducible results
    random.seed(123)
    np.random.seed(123)
    
    # Generate a 2D random walk
    steps = 2000
    x_original, y_original = random_walk_2d(steps)
    
    # Perform loop erasure
    x_erased, y_erased = loop_erased_random_walk(x_original, y_original)
    
    # Get convex hull (outer boundary)
    try:
        # Try to import scipy for convex hull
        from scipy.spatial import ConvexHull
        hull_x, hull_y = get_convex_hull(x_original, y_original)
        has_scipy = True
    except ImportError:
        # Fallback: create approximate boundary using min/max points
        has_scipy = False
        min_x, max_x = min(x_original), max(x_original)
        min_y, max_y = min(y_original), max(y_original)
        
        # Find points closest to corners
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        boundary_points = []
        
        for corner_x, corner_y in corners:
            distances = [(x - corner_x)**2 + (y - corner_y)**2 
                        for x, y in zip(x_original, y_original)]
            closest_idx = distances.index(min(distances))
            boundary_points.append((x_original[closest_idx], y_original[closest_idx]))
        
        # Close the boundary
        boundary_points.append(boundary_points[0])
        hull_x, hull_y = zip(*boundary_points)
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Plot original walk path (light gray, thin line)
    plt.plot(x_original, y_original, color='lightgray', linewidth=0.5, alpha=0.7, 
             label=f'Original Walk ({len(x_original)} steps)')
    
    # Plot convex hull / outer boundary (blue, thick line)
    boundary_label = 'Convex Hull' if has_scipy else 'Approximate Boundary'
    plt.plot(hull_x, hull_y, color='blue', linewidth=3, alpha=0.8, 
             label=boundary_label)
    
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
Boundary method: {boundary_label}"""
    
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
    
    return x_original, y_original, x_erased, y_erased, hull_x, hull_y

if __name__ == "__main__":
    create_loop_erased_visualization()