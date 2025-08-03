import numpy as np
import matplotlib.pyplot as plt
import random
import os
from collections import deque

class SelfAvoidingWalk:
    """
    Self-avoiding walk (SAW) implementation on Z² lattice.
    Uses backtracking algorithm to generate walks without self-intersections.
    """
    
    def __init__(self):
        self.moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        
    def generate_saw(self, target_length, max_attempts=500):
        """
        Generate a self-avoiding walk using simple backtracking.
        Optimized for shorter walks (< 300 steps).
        """
        
        for attempt in range(max_attempts):
            path = [(0, 0)]
            visited = {(0, 0)}
            
            while len(path) < target_length + 1:
                current_x, current_y = path[-1]
                
                # Find available moves
                available_moves = []
                for dx, dy in self.moves:
                    next_pos = (current_x + dx, current_y + dy)
                    if next_pos not in visited:
                        available_moves.append(next_pos)
                
                if not available_moves:
                    # Dead end - restart (simple strategy)
                    break
                else:
                    # Choose random available move
                    next_pos = random.choice(available_moves)
                    path.append(next_pos)
                    visited.add(next_pos)
            
            if len(path) >= target_length + 1:
                return path
        
        return None
    
    def generate_saw_pivot(self, target_length, max_attempts=10000):
        """
        Generate SAW using pivot algorithm (more efficient for longer walks).
        """
        # Start with a simple walk
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]  # Simple initial path
        
        for attempt in range(max_attempts):
            if len(path) >= target_length + 1:
                return path[:target_length + 1]
            
            # Try to extend the walk
            current_x, current_y = path[-1]
            visited = set(path)
            
            available_moves = []
            for dx, dy in self.moves:
                next_pos = (current_x + dx, current_y + dy)
                if next_pos not in visited:
                    available_moves.append(next_pos)
            
            if available_moves:
                next_pos = random.choice(available_moves)
                path.append(next_pos)
            else:
                # Try pivot move
                if len(path) > 4:
                    self.attempt_pivot(path)
                else:
                    # Restart with longer initial segment
                    path = self.generate_simple_start(min(10, target_length))
        
        return path if len(path) >= 4 else None
    
    def generate_simple_start(self, length):
        """Generate a simple non-intersecting starting path."""
        path = [(0, 0)]
        direction = 0  # 0=E, 1=N, 2=W, 3=S
        
        for i in range(length):
            if i > 0 and i % 5 == 0:  # Change direction every 5 steps
                direction = (direction + 1) % 4
            
            dx, dy = self.moves[direction]
            last_x, last_y = path[-1]
            path.append((last_x + dx, last_y + dy))
        
        return path
    
    def attempt_pivot(self, path):
        """Attempt a pivot move to extend the walk."""
        if len(path) < 3:
            return False
        
        # Choose a random pivot point (not the ends)
        pivot_idx = random.randint(1, len(path) - 2)
        
        # Try rotating the tail around the pivot
        pivot_x, pivot_y = path[pivot_idx]
        
        # Simple 90-degree rotation
        for rotation in [1, -1]:  # Try clockwise and counterclockwise
            new_tail = []
            valid = True
            
            for i in range(pivot_idx + 1, len(path)):
                old_x, old_y = path[i]
                # Rotate around pivot
                rel_x, rel_y = old_x - pivot_x, old_y - pivot_y
                if rotation == 1:  # Clockwise
                    new_rel_x, new_rel_y = rel_y, -rel_x
                else:  # Counterclockwise
                    new_rel_x, new_rel_y = -rel_y, rel_x
                
                new_pos = (pivot_x + new_rel_x, pivot_y + new_rel_y)
                
                # Check if new position conflicts with existing path
                if new_pos in path[:pivot_idx + 1] or new_pos in new_tail:
                    valid = False
                    break
                
                new_tail.append(new_pos)
            
            if valid and new_tail:
                # Apply the pivot
                path[pivot_idx + 1:] = new_tail
                return True
        
        return False
    
    def analyze_saw_properties(self, path):
        """Analyze geometric properties of the SAW."""
        if not path or len(path) < 2:
            return {}
        
        # End-to-end distance
        start_x, start_y = path[0]
        end_x, end_y = path[-1]
        end_to_end = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Radius of gyration
        center_x = np.mean([pos[0] for pos in path])
        center_y = np.mean([pos[1] for pos in path])
        
        radius_gyration = np.sqrt(np.mean([
            (pos[0] - center_x)**2 + (pos[1] - center_y)**2 
            for pos in path
        ]))
        
        # Bounding box
        xs = [pos[0] for pos in path]
        ys = [pos[1] for pos in path]
        bbox_width = max(xs) - min(xs)
        bbox_height = max(ys) - min(ys)
        
        return {
            'length': len(path) - 1,  # Number of steps
            'end_to_end': end_to_end,
            'radius_gyration': radius_gyration,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_area': bbox_width * bbox_height
        }

def create_saw_visualization():
    """Create visualizations of self-avoiding walks."""
    
    random.seed(123)
    np.random.seed(123)
    
    saw = SelfAvoidingWalk()
    
    # Generate SAWs of different lengths (shorter for efficiency)
    lengths = [25, 50, 75, 100]
    walks = []
    
    print("Generating self-avoiding walks...")
    for length in lengths:
        print(f"Generating SAW of length {length}...")
        path = saw.generate_saw(length, max_attempts=5000)
        if path:
            walks.append((length, path))
            print(f"Success! Generated {len(path)-1} steps")
        else:
            print(f"Failed to generate full length {length}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (length, path) in enumerate(walks[:4]):
        ax = axes[i]
        
        if path:
            xs = [pos[0] for pos in path]
            ys = [pos[1] for pos in path]
            
            # Plot the walk
            ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.8)
            ax.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20, alpha=0.7)
            
            # Mark start and end
            ax.scatter(xs[0], ys[0], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(xs[-1], ys[-1], c='red', s=100, marker='s', label='End', zorder=5)
            
            # Analyze properties
            props = saw.analyze_saw_properties(path)
            
            ax.set_title(f'Self-Avoiding Walk (N = {props["length"]})\n'
                        f'End-to-end: {props["end_to_end"]:.1f}, '
                        f'R_g: {props["radius_gyration"]:.1f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('images/self_avoiding_walks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return walks

def analyze_saw_scaling():
    """
    Analyze scaling properties of self-avoiding walks.
    Expected: R ~ N^ν where ν ≈ 3/4 = 0.75 in 2D.
    """
    
    print("Starting SAW scaling analysis...")
    
    random.seed(42)
    np.random.seed(42)
    
    saw = SelfAvoidingWalk()
    
    # Test different walk lengths (optimized for efficiency)
    test_lengths = [15, 20, 30, 40, 50, 70, 90]
    
    results = {
        'lengths': [],
        'end_to_end': [],
        'radius_gyration': [],
        'bbox_areas': []
    }
    
    for length in test_lengths:
        print(f"Analyzing SAW length {length}...")
        
        length_end_to_end = []
        length_radius_gyration = []
        length_bbox_areas = []
        
        # Multiple runs for statistics
        successful_runs = 0
        max_runs = 10
        
        for run in range(max_runs):
            path = saw.generate_saw(length, max_attempts=2000)
            
            if path and len(path) >= length * 0.8:  # Accept if we got close
                props = saw.analyze_saw_properties(path)
                length_end_to_end.append(props['end_to_end'])
                length_radius_gyration.append(props['radius_gyration'])
                length_bbox_areas.append(props['bbox_area'])
                successful_runs += 1
        
        if successful_runs >= 3:  # Need at least 3 successful runs
            results['lengths'].append(length)
            results['end_to_end'].append(np.mean(length_end_to_end))
            results['radius_gyration'].append(np.mean(length_radius_gyration))
            results['bbox_areas'].append(np.mean(length_bbox_areas))
            
            print(f"  Success rate: {successful_runs}/{max_runs}")
            print(f"  Average end-to-end: {np.mean(length_end_to_end):.2f}")
            print(f"  Average R_g: {np.mean(length_radius_gyration):.2f}")
    
    # Create scaling plots
    if len(results['lengths']) >= 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        lengths = np.array(results['lengths'])
        end_to_end = np.array(results['end_to_end'])
        radius_gyration = np.array(results['radius_gyration'])
        
        # Plot 1: End-to-end distance scaling
        ax1.loglog(lengths, end_to_end, 'ro-', label='End-to-End Distance')
        ax1.loglog(lengths, radius_gyration, 'bo-', label='Radius of Gyration')
        
        # Fit power laws
        log_lengths = np.log(lengths)
        
        # End-to-end fit
        log_end_to_end = np.log(end_to_end)
        fit_e2e = np.polyfit(log_lengths, log_end_to_end, 1)
        nu_e2e = fit_e2e[0]
        
        # Radius of gyration fit
        log_rg = np.log(radius_gyration)
        fit_rg = np.polyfit(log_lengths, log_rg, 1)
        nu_rg = fit_rg[0]
        
        # Plot fits
        ax1.loglog(lengths, np.exp(fit_e2e[1]) * lengths**fit_e2e[0], 'r--', 
                  label=f'End-to-End: ν = {nu_e2e:.3f}')
        ax1.loglog(lengths, np.exp(fit_rg[1]) * lengths**fit_rg[0], 'b--',
                  label=f'R_g: ν = {nu_rg:.3f}')
        
        # Theoretical scaling
        ax1.loglog(lengths, lengths**(3/4) * (end_to_end[0] / lengths[0]**(3/4)), 'k--',
                  alpha=0.7, label='Theoretical: N^{3/4}')
        
        ax1.set_xlabel('Walk Length N')
        ax1.set_ylabel('Distance')
        ax1.set_title(f'SAW Scaling Analysis\n'
                     f'End-to-End: ν = {nu_e2e:.3f}, R_g: ν = {nu_rg:.3f}\n'
                     f'(Theoretical: ν = 3/4 = 0.75)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scaling collapse
        ax2.loglog(lengths, end_to_end / (lengths**nu_e2e), 'ro-', 
                  label=f'Rescaled End-to-End (ν={nu_e2e:.3f})')
        ax2.loglog(lengths, radius_gyration / (lengths**nu_rg), 'bo-',
                  label=f'Rescaled R_g (ν={nu_rg:.3f})')
        
        ax2.set_xlabel('Walk Length N')
        ax2.set_ylabel('R / N^ν')
        ax2.set_title('Scaling Collapse')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/saw_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nScaling Results:")
        print(f"End-to-end distance: R ~ N^{nu_e2e:.3f} (theoretical: N^0.75)")
        print(f"Radius of gyration: R_g ~ N^{nu_rg:.3f} (theoretical: N^0.75)")
        print(f"Critical exponent ν ≈ {(nu_e2e + nu_rg)/2:.3f} (theoretical: 3/4 = 0.75)")
        
        return nu_e2e, nu_rg
    else:
        print("Not enough successful runs for scaling analysis")
        return None, None

if __name__ == "__main__":
    print("Self-Avoiding Walk Analysis on Z²")
    print("=" * 50)
    
    # Create images directory
    os.makedirs('images', exist_ok=True)
    
    # Generate visualizations
    walks = create_saw_visualization()
    print(f"Generated {len(walks)} self-avoiding walks")
    print("Visualization saved: images/self_avoiding_walks.png")
    
    # Scaling analysis
    print(f"\n" + "="*50)
    nu_e2e, nu_rg = analyze_saw_scaling()
    
    if nu_e2e is not None:
        print("Scaling analysis saved: images/saw_scaling_analysis.png")
    
    print(f"\nAnalysis complete!")
    print(f"Self-avoiding walks should have critical exponent ν = 3/4 = 0.75 in 2D")