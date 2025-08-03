import numpy as np
import matplotlib.pyplot as plt
import random
import os
from collections import deque, defaultdict

class HexagonalLattice:
    """
    Hexagonal lattice implementation for percolation studies.
    Uses axial coordinates (q, r) for hexagonal grid.
    """
    
    def __init__(self, size):
        self.size = size
        self.grid = {}  # (q, r) -> occupied status
        self.pc = 0.5   # Critical percolation probability for hexagonal lattice
        
    def get_neighbors(self, q, r):
        """Get the 6 neighbors of a hexagonal cell in axial coordinates."""
        # Hexagonal neighbors in axial coordinates
        directions = [
            (1, 0), (1, -1), (0, -1),
            (-1, 0), (-1, 1), (0, 1)
        ]
        neighbors = []
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if self.is_valid_cell(nq, nr):
                neighbors.append((nq, nr))
        return neighbors
    
    def is_valid_cell(self, q, r):
        """Check if cell coordinates are within the lattice bounds."""
        # Use rhombus shape for simplicity
        return abs(q) <= self.size and abs(r) <= self.size and abs(q + r) <= self.size
    
    def generate_percolation(self, p=None):
        """Generate a percolation configuration with probability p."""
        if p is None:
            p = self.pc
            
        self.grid = {}
        for q in range(-self.size, self.size + 1):
            for r in range(-self.size, self.size + 1):
                if self.is_valid_cell(q, r):
                    self.grid[(q, r)] = random.random() < p
        
        return self.grid
    
    def axial_to_cartesian(self, q, r):
        """Convert axial coordinates to cartesian for visualization."""
        x = q + r / 2
        y = r * np.sqrt(3) / 2
        return x, y
    
    def find_clusters(self):
        """Find all connected clusters using flood fill."""
        visited = set()
        clusters = []
        
        for cell in self.grid:
            if cell not in visited and self.grid[cell]:
                cluster = []
                queue = deque([cell])
                visited.add(cell)
                
                while queue:
                    current = queue.popleft()
                    cluster.append(current)
                    
                    for neighbor in self.get_neighbors(*current):
                        if neighbor not in visited and self.grid.get(neighbor, False):
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                clusters.append(cluster)
        
        return clusters
    
    def find_percolating_cluster(self):
        """Find the percolating cluster that spans the lattice."""
        clusters = self.find_clusters()
        
        for cluster in clusters:
            # Check if cluster spans from one side to another
            q_coords = [cell[0] for cell in cluster]
            r_coords = [cell[1] for cell in cluster]
            
            # Check if it spans significant portion of lattice
            q_span = max(q_coords) - min(q_coords)
            r_span = max(r_coords) - min(r_coords)
            
            if q_span > self.size * 0.7 or r_span > self.size * 0.7:
                return cluster
        
        return None
    
    def find_separating_path(self, percolating_cluster):
        """
        Find the separating path - the boundary between the percolating cluster
        and the exterior, which has fractal dimension 7/4 at criticality.
        """
        if not percolating_cluster:
            return []
        
        cluster_set = set(percolating_cluster)
        boundary_cells = []
        
        # Find boundary cells - occupied cells with at least one unoccupied neighbor
        for cell in percolating_cluster:
            q, r = cell
            for neighbor in self.get_neighbors(q, r):
                if neighbor not in cluster_set or not self.grid.get(neighbor, False):
                    boundary_cells.append(cell)
                    break
        
        # Trace the separating path around the cluster
        if not boundary_cells:
            return []
        
        # Use a simple boundary tracing algorithm
        separating_path = self.trace_boundary(cluster_set, boundary_cells)
        
        return separating_path
    
    def trace_boundary(self, cluster_set, boundary_cells):
        """
        Trace the boundary path around the percolating cluster.
        This gives us the separating path.
        """
        if not boundary_cells:
            return []
        
        # Start from the leftmost boundary cell
        start_cell = min(boundary_cells, key=lambda cell: (cell[0], cell[1]))
        
        # Use a simplified boundary following approach
        # For now, return all boundary cells sorted by position
        # A more sophisticated boundary tracing algorithm could be implemented
        
        boundary_path = list(set(boundary_cells))
        
        # Sort by angle from center to create a reasonable path
        center_q = sum(cell[0] for cell in cluster_set) / len(cluster_set)
        center_r = sum(cell[1] for cell in cluster_set) / len(cluster_set)
        
        def angle_from_center(cell):
            q, r = cell
            x, y = self.axial_to_cartesian(q - center_q, r - center_r)
            return np.arctan2(y, x)
        
        boundary_path.sort(key=angle_from_center)
        
        return boundary_path

def create_percolation_visualization(size=50):
    """
    Create a visualization of critical percolation on hexagonal lattice
    with the separating path highlighted.
    """
    
    # Set seed for reproducible results
    random.seed(123)
    np.random.seed(123)
    
    # Create hexagonal lattice
    lattice = HexagonalLattice(size)
    
    # Generate critical percolation
    print(f"Generating critical percolation on hexagonal lattice (size={size})...")
    lattice.generate_percolation(p=lattice.pc)
    
    # Find percolating cluster
    print("Finding percolating cluster...")
    percolating_cluster = lattice.find_percolating_cluster()
    
    if percolating_cluster:
        print(f"Found percolating cluster with {len(percolating_cluster)} sites")
        
        # Find separating path
        print("Tracing separating path...")
        separating_path = lattice.find_separating_path(percolating_cluster)
        print(f"Separating path has {len(separating_path)} boundary sites")
    else:
        print("No percolating cluster found")
        separating_path = []
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Full percolation configuration
    occupied_x, occupied_y = [], []
    empty_x, empty_y = [], []
    
    for (q, r), occupied in lattice.grid.items():
        x, y = lattice.axial_to_cartesian(q, r)
        if occupied:
            occupied_x.append(x)
            occupied_y.append(y)
        else:
            empty_x.append(x)
            empty_y.append(y)
    
    # Plot empty sites
    ax1.scatter(empty_x, empty_y, c='lightgray', s=20, alpha=0.3, marker='h')
    
    # Plot occupied sites
    ax1.scatter(occupied_x, occupied_y, c='black', s=25, alpha=0.8, marker='h')
    
    # Highlight percolating cluster
    if percolating_cluster:
        perc_x, perc_y = [], []
        for q, r in percolating_cluster:
            x, y = lattice.axial_to_cartesian(q, r)
            perc_x.append(x)
            perc_y.append(y)
        
        ax1.scatter(perc_x, perc_y, c='red', s=30, alpha=0.9, marker='h',
                   label=f'Percolating Cluster ({len(percolating_cluster)} sites)')
    
    ax1.set_title(f'Critical Percolation on Hexagonal Lattice\n(p = {lattice.pc}, size = {size})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Focus on separating path
    if percolating_cluster and separating_path:
        # Plot the percolating cluster
        ax2.scatter(perc_x, perc_y, c='lightblue', s=25, alpha=0.6, marker='h',
                   label=f'Percolating Cluster')
        
        # Plot separating path
        sep_x, sep_y = [], []
        for q, r in separating_path:
            x, y = lattice.axial_to_cartesian(q, r)
            sep_x.append(x)
            sep_y.append(y)
        
        ax2.scatter(sep_x, sep_y, c='red', s=40, alpha=1.0, marker='h',
                   label=f'Separating Path ({len(separating_path)} sites)')
        
        # Connect separating path points
        if len(separating_path) > 1:
            ax2.plot(sep_x + [sep_x[0]], sep_y + [sep_y[0]], 'r-', alpha=0.7, linewidth=2)
        
        ax2.set_title(f'Separating Path (Fractal Dimension ≈ 7/4 = 1.75)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No percolating cluster found\n(try different random seed)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Separating Path')
    
    plt.tight_layout()
    plt.savefig('images/hexagonal_percolation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return lattice, percolating_cluster, separating_path

def analyze_separating_path_scaling():
    """
    Analyze the scaling of separating path length with system size
    to estimate the fractal dimension (should be ≈ 7/4 = 1.75).
    """
    
    print("Starting separating path scaling analysis...")
    
    # Test different system sizes
    sizes = [20, 30, 40, 50, 70, 90, 120]
    path_lengths = []
    perimeter_lengths = []
    cluster_sizes = []
    
    random.seed(42)
    np.random.seed(42)
    
    for size in sizes:
        print(f"Analyzing size {size}...")
        
        size_path_lengths = []
        size_perimeter_lengths = []
        size_cluster_sizes = []
        
        # Multiple runs for each size
        for run in range(10):
            lattice = HexagonalLattice(size)
            lattice.generate_percolation(p=lattice.pc)
            
            percolating_cluster = lattice.find_percolating_cluster()
            
            if percolating_cluster:
                separating_path = lattice.find_separating_path(percolating_cluster)
                
                size_path_lengths.append(len(separating_path))
                size_cluster_sizes.append(len(percolating_cluster))
                
                # Calculate perimeter of cluster
                cluster_set = set(percolating_cluster)
                perimeter = 0
                for q, r in percolating_cluster:
                    neighbors = lattice.get_neighbors(q, r)
                    perimeter += sum(1 for neighbor in neighbors 
                                   if neighbor not in cluster_set or not lattice.grid.get(neighbor, False))
                size_perimeter_lengths.append(perimeter)
        
        if size_path_lengths:
            path_lengths.append(np.mean(size_path_lengths))
            perimeter_lengths.append(np.mean(size_perimeter_lengths))
            cluster_sizes.append(np.mean(size_cluster_sizes))
        else:
            path_lengths.append(0)
            perimeter_lengths.append(0)
            cluster_sizes.append(0)
    
    # Create scaling plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter out zero values
    valid_indices = [i for i, length in enumerate(path_lengths) if length > 0]
    valid_sizes = [sizes[i] for i in valid_indices]
    valid_path_lengths = [path_lengths[i] for i in valid_indices]
    valid_cluster_sizes = [cluster_sizes[i] for i in valid_indices]
    
    if len(valid_sizes) >= 3:
        # Plot 1: Path length vs system size
        ax1.loglog(valid_sizes, valid_path_lengths, 'ro-', label='Separating Path Length')
        
        # Fit power law
        log_sizes = np.log(valid_sizes)
        log_lengths = np.log(valid_path_lengths)
        fit = np.polyfit(log_sizes, log_lengths, 1)
        exponent = fit[0]
        
        # Plot theoretical scaling
        ax1.loglog(valid_sizes, np.exp(fit[1]) * np.array(valid_sizes)**fit[0], 'r--', 
                  label=f'Fit: L ∝ N^{{{exponent:.3f}}}')
        ax1.loglog(valid_sizes, np.array(valid_sizes)**(7/4) * (valid_path_lengths[0] / valid_sizes[0]**(7/4)), 'k--',
                  label='Theoretical: L ∝ N^{7/4}', alpha=0.7)
        
        ax1.set_xlabel('System Size N')
        ax1.set_ylabel('Separating Path Length')
        ax1.set_title(f'Separating Path Scaling\n(Measured exponent: {exponent:.3f}, Theoretical: 1.75)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Path length vs cluster size
        ax2.loglog(valid_cluster_sizes, valid_path_lengths, 'bo-', label='Path vs Cluster Size')
        
        # Fit
        log_cluster = np.log(valid_cluster_sizes)
        fit2 = np.polyfit(log_cluster, log_lengths, 1)
        exponent2 = fit2[0]
        
        ax2.loglog(valid_cluster_sizes, np.exp(fit2[1]) * np.array(valid_cluster_sizes)**fit2[0], 'b--',
                  label=f'Fit: L ∝ A^{{{exponent2:.3f}}}')
        
        ax2.set_xlabel('Cluster Size (Area)')
        ax2.set_ylabel('Separating Path Length')
        ax2.set_title(f'Path Length vs Cluster Area\n(Exponent: {exponent2:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        print(f"\nScaling Results:")
        print(f"Separating path vs system size: L ∝ N^{exponent:.3f} (theoretical: N^{7/4:.3f})")
        print(f"Separating path vs cluster area: L ∝ A^{exponent2:.3f}")
        print(f"Estimated fractal dimension: D ≈ {exponent:.3f} (theoretical: 7/4 = 1.75)")
    
    plt.tight_layout()
    plt.savefig('images/separating_path_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return valid_sizes, valid_path_lengths

if __name__ == "__main__":
    print("Critical Percolation on Hexagonal Lattice with Separating Path Analysis")
    print("=" * 70)
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Generate visualization
    lattice, cluster, path = create_percolation_visualization(size=60)
    
    if cluster:
        print(f"\nVisualization created:")
        print(f"- Percolating cluster: {len(cluster)} sites")
        print(f"- Separating path: {len(path)} boundary sites")
        print(f"- Image saved: images/hexagonal_percolation.png")
    
    # Scaling analysis
    print(f"\n" + "="*70)
    sizes, path_lengths = analyze_separating_path_scaling()
    print(f"- Scaling analysis saved: images/separating_path_scaling.png")
    
    print(f"\nAnalysis complete!")
    print(f"The separating path should have fractal dimension 7/4 = 1.75 at criticality.")