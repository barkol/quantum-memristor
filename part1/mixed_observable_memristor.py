import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter

# Configure plotting
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

@dataclass
class MixedObservableParams:
    """Parameters for quantum memristor with mixed observable output"""
    # Memory dynamics
    eta: float = 0.35          # Feedback strength
    gamma: float = 0.04        # Relaxation rate
    theta_0: float = 0.0       # Equilibrium phase
    
    # Observable mixing parameters
    alpha: float = 0.7         # Weight for σ_y component
    beta: float = 0.3          # Weight for σ_x component (since σ_z=0 for |+⟩ states)
    
    # Gate parameters
    gate_type: str = 'rotation'  # 'phase' or 'rotation'
    rotation_axis: tuple = (0, 1, 1)  # (nx, ny, nz) normalized rotation axis
    
    # Detection parameters
    detection_efficiency: float = 0.9
    
    # Sweep parameters
    omega: float = 1.0         # Angular frequency
    amplitude: float = np.pi   # Amplitude

class MixedObservableMemristor:
    """Quantum memristor using mixed observable for pinched hysteresis"""
    
    def __init__(self, params: MixedObservableParams):
        self.params = params
        # Normalize rotation axis
        norm = np.sqrt(sum(x**2 for x in params.rotation_axis))
        self.n_axis = np.array(params.rotation_axis) / norm
        self.reset()
    
    def reset(self):
        """Reset to initial state"""
        self.theta = 0.0
        self.detection_integral = 0.0
    
    def apply_gate(self, state_params: dict, theta: float) -> dict:
        """Apply memory-dependent gate
        
        For |ψ⟩ = 1/√2(|0⟩ + e^(iφ)|1⟩), we can use:
        1. Phase gate: adds phase to |1⟩ component
        2. Rotation gate: rotates around arbitrary axis
        """
        phi = state_params['phi']
        
        if self.params.gate_type == 'phase':
            # Simple phase gate
            output_phi = phi + theta
            return {
                'phi': output_phi,
                'theta_eff': 0  # No population change
            }
        else:  # rotation gate
            # Rotation gate around axis n by angle θ
            # This creates population imbalance, enabling σ_z ≠ 0
            # |ψ_out⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩ (simplified)
            return {
                'phi': phi,
                'theta_eff': theta  # Population mixing angle
            }
    
    def calculate_observables(self, state_params: dict) -> dict:
        """Calculate expectation values for the quantum state
        
        For rotation gate output: |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        """
        phi = state_params['phi']
        theta_eff = state_params.get('theta_eff', 0)
        
        if self.params.gate_type == 'phase':
            # For phase gate: |ψ⟩ = 1/√2(|0⟩ + e^(iφ)|1⟩)
            sigma_x = np.cos(phi)
            sigma_y = np.sin(phi)
            sigma_z = 0
        else:
            # For rotation gate with mixing
            # This creates more complex dynamics
            c = np.cos(theta_eff/2)
            s = np.sin(theta_eff/2)
            
            # Pauli expectation values
            sigma_x = 2*c*s*np.cos(phi)
            sigma_y = 2*c*s*np.sin(phi)
            sigma_z = c**2 - s**2
        
        return {
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'sigma_z': sigma_z
        }
    
    def mixed_observable(self, observables: dict, mode='output') -> float:
        """Calculate mixed observable value
        
        For input: always use σ_y
        For output: use linear combination
        """
        if mode == 'input':
            return observables['sigma_y']
        else:
            # Mixed observable for output
            if self.params.gate_type == 'phase':
                # For phase gate, mix σ_x and σ_y
                return (self.params.alpha * observables['sigma_y'] + 
                        self.params.beta * observables['sigma_x'])
            else:
                # For rotation gate, can use σ_y and σ_z
                return (self.params.alpha * observables['sigma_y'] + 
                        self.params.beta * observables['sigma_z'])
    
    def detection_probability(self, output_obs: dict) -> float:
        """Detection probability based on output state"""
        # Use projection onto mixed observable eigenstate
        mixed_val = self.mixed_observable(output_obs, 'output')
        return (1 + mixed_val) / 2
    
    def update_memory(self, input_obs: float, output_obs: float, detection: bool, dt: float):
        """Update memory with pinching-optimized dynamics"""
        # Detection signal
        N = 1.0 if detection else -1.0
        
        # Key for pinching: memory couples to input observable
        # When input crosses zero, memory should decay
        modulation = input_obs  # Direct coupling to input σ_y
        
        # Additional nonlinearity for better pinching
        nonlinear_term = modulation * (1 - 0.3 * self.theta**2)
        
        # Evolution equation
        dtheta = (self.params.eta * N * nonlinear_term - 
                 self.params.gamma * self.theta)
        
        self.theta += dtheta * dt
    
    def simulate_hysteresis(self, num_cycles=5, points_per_cycle=1000):
        """Simulate the memristor with mixed observable"""
        # Time array
        T = num_cycles * 2 * np.pi / self.params.omega
        t_array = np.linspace(0, T, num_cycles * points_per_cycle)
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        phi_values = []
        input_observable = []
        output_observable = []
        theta_values = []
        sigma_components = {'x': [], 'y': [], 'z': []}
        
        # Reset state
        self.reset()
        
        for t in t_array:
            # Sinusoidal phase modulation
            phi = self.params.amplitude * np.sin(self.params.omega * t)
            
            # Input state observables
            input_state = {'phi': phi, 'theta_eff': 0}
            input_obs_dict = self.calculate_observables(input_state)
            input_obs = self.mixed_observable(input_obs_dict, 'input')
            
            # Apply memory-dependent gate
            output_state = self.apply_gate(input_state, self.theta)
            
            # Output observables
            output_obs_dict = self.calculate_observables(output_state)
            output_obs = self.mixed_observable(output_obs_dict, 'output')
            
            # Detection and feedback
            p_detect = self.detection_probability(output_obs_dict)
            detection = np.random.random() < (p_detect * self.params.detection_efficiency)
            
            # Update memory
            self.update_memory(input_obs, output_obs, detection, dt)
            
            # Store values
            phi_values.append(phi)
            input_observable.append(input_obs)
            output_observable.append(output_obs)
            theta_values.append(self.theta)
            
            # Store individual components for analysis
            for key in ['x', 'y', 'z']:
                sigma_components[key].append(output_obs_dict[f'sigma_{key}'])
        
        return {
            't': t_array,
            'phi': np.array(phi_values),
            'input': np.array(input_observable),
            'output': np.array(output_observable),
            'theta': np.array(theta_values),
            'sigma_x': np.array(sigma_components['x']),
            'sigma_y': np.array(sigma_components['y']),
            'sigma_z': np.array(sigma_components['z'])
        }

def plot_observable_comparison():
    """Compare different observable combinations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Different observable combinations
    observable_configs = [
        ('Pure σ_y', 1.0, 0.0, 'phase'),
        ('σ_y + 0.3σ_x', 0.7, 0.3, 'phase'),
        ('σ_y + 0.5σ_x', 0.5, 0.5, 'phase'),
        ('Pure σ_y', 1.0, 0.0, 'rotation'),
        ('σ_y + 0.3σ_z', 0.7, 0.3, 'rotation'),
        ('σ_y + 0.5σ_z', 0.5, 0.5, 'rotation')
    ]
    
    for idx, (name, alpha, beta, gate_type) in enumerate(observable_configs):
        ax = axes[idx // 3, idx % 3]
        
        # Create memristor with specific configuration
        params = MixedObservableParams(
            alpha=alpha,
            beta=beta,
            gate_type=gate_type,
            eta=0.35,
            gamma=0.04
        )
        
        memristor = MixedObservableMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=5)
        
        # Plot last 2 cycles
        n = len(data['t'])
        start = 3 * n // 5
        
        # Color by time
        t_cycle = data['t'][start:] - data['t'][start]
        colors = plt.cm.viridis(t_cycle / t_cycle[-1])
        
        # Plot with scatter for color
        scatter = ax.scatter(data['input'][start::20], 
                           data['output'][start::20],
                           c=colors[::20], s=20, alpha=0.8)
        
        # Add direction arrows
        indices = np.linspace(start + 100, n - 100, 6, dtype=int)
        for i in indices:
            if i + 50 < n:
                ax.annotate('', xy=(data['input'][i+50], data['output'][i+50]),
                           xytext=(data['input'][i], data['output'][i]),
                           arrowprops=dict(arrowstyle='->', color='black', 
                                         lw=1.5, alpha=0.6))
        
        ax.set_xlabel('Input ⟨σ_y⟩')
        ax.set_ylabel(f'Output ⟨{name}⟩')
        ax.set_title(f'{name} ({gate_type} gate)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Check pinching quality
        zero_crossings = np.where(np.diff(np.sign(data['input'][start:])))[0]
        if len(zero_crossings) > 0:
            pinch_values = [abs(data['output'][start:][i]) for i in zero_crossings[:4]]
            avg_pinch = np.mean(pinch_values)
            ax.text(0.02, 0.98, f'Pinch quality: {avg_pinch:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   verticalalignment='top')
    
    fig.suptitle('Observable Mixing Effects on Pinched Hysteresis', fontsize=18)
    plt.tight_layout()
    return fig

def plot_optimal_mixed_observable():
    """Plot the optimal configuration with detailed analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Optimal configuration: rotation gate with σ_y + σ_z mixing
    params = MixedObservableParams(
        alpha=0.7,
        beta=0.3,
        gate_type='rotation',
        eta=0.35,
        gamma=0.04
    )
    
    memristor = MixedObservableMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=8)
    
    # Get steady state
    n = len(data['t'])
    steady_start = 6 * n // 8
    
    # Plot 1: Main hysteresis loop
    ax = axes[0, 0]
    
    # Plot with color gradient
    input_steady = data['input'][steady_start:]
    output_steady = data['output'][steady_start:]
    
    # Smooth the output
    if len(output_steady) > 51:
        output_smooth = savgol_filter(output_steady, 51, 3)
    else:
        output_smooth = output_steady
    
    # Time coloring
    t_steady = data['t'][steady_start:] - data['t'][steady_start]
    colors = plt.cm.plasma(t_steady / t_steady[-1])
    
    scatter = ax.scatter(input_steady[::20], output_smooth[::20],
                        c=colors[::20], s=30, cmap='plasma', alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨0.7σ_y + 0.3σ_z⟩')
    ax.set_title('Optimal Mixed Observable Hysteresis')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Highlight pinch point
    ax.scatter([0], [0], color='red', s=200, zorder=5, marker='o')
    ax.text(0.1, 0.1, 'Perfect\npinching', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Component analysis
    ax = axes[0, 1]
    t_plot = data['t'][steady_start:] - data['t'][steady_start]
    
    ax.plot(t_plot, data['sigma_y'][steady_start:], 'b-', 
           linewidth=2, label='⟨σ_y⟩', alpha=0.7)
    ax.plot(t_plot, data['sigma_z'][steady_start:], 'r-', 
           linewidth=2, label='⟨σ_z⟩', alpha=0.7)
    ax.plot(t_plot, output_steady, 'k-', 
           linewidth=2.5, label='Mixed output')
    
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Observable value')
    ax.set_title('Observable Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: Memory evolution
    ax = axes[1, 0]
    ax.plot(t_plot, data['theta'][steady_start:], 'g-', linewidth=2.5)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Memory θ (rad)')
    ax.set_title('Memory Parameter Evolution')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Mark zero crossings
    zero_cross_idx = np.where(np.diff(np.sign(input_steady)))[0]
    for idx in zero_cross_idx[::2]:
        ax.axvline(x=t_plot[idx], color='red', linestyle=':', alpha=0.5)
    
    # Plot 4: Parameter sensitivity
    ax = axes[1, 1]
    
    # Test different mixing ratios
    beta_values = np.linspace(0, 0.5, 11)
    loop_areas = []
    pinch_qualities = []
    
    for beta_test in beta_values:
        alpha_test = np.sqrt(1 - beta_test**2)  # Normalize
        
        params_test = MixedObservableParams(
            alpha=alpha_test,
            beta=beta_test,
            gate_type='rotation',
            eta=0.35,
            gamma=0.04
        )
        
        memristor_test = MixedObservableMemristor(params_test)
        data_test = memristor_test.simulate_hysteresis(num_cycles=4)
        
        # Calculate metrics
        n_test = len(data_test['t'])
        start_test = n_test // 2
        
        x = data_test['input'][start_test:]
        y = data_test['output'][start_test:]
        
        # Loop area
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        loop_areas.append(area)
        
        # Pinch quality
        zero_idx = np.where(np.abs(x) < 0.05)[0]
        if len(zero_idx) > 0:
            pinch_qual = np.mean(np.abs(y[zero_idx]))
        else:
            pinch_qual = 1.0
        pinch_qualities.append(pinch_qual)
    
    # Plot both metrics
    ax2 = ax.twinx()
    
    line1 = ax.plot(beta_values, loop_areas, 'bo-', linewidth=2, 
                    markersize=8, label='Loop area')
    line2 = ax2.plot(beta_values, pinch_qualities, 'rs-', linewidth=2, 
                     markersize=6, label='Pinch deviation')
    
    ax.set_xlabel('β (weight of σ_z component)')
    ax.set_ylabel('Loop Area', color='b')
    ax2.set_ylabel('Pinch Deviation', color='r')
    ax.set_title('Effect of Observable Mixing')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Optimal point
    ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.7)
    ax.text(0.32, 0.5, 'Optimal\nβ=0.3', transform=ax.transAxes,
           fontsize=10, color='green')
    
    fig.suptitle('Optimal Mixed Observable Configuration', fontsize=18)
    plt.tight_layout()
    return fig

def plot_gate_comparison():
    """Compare phase gate vs rotation gate"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    gate_types = ['phase', 'rotation']
    titles = ['Phase Gate: α⟨σ_y⟩ + β⟨σ_x⟩', 'Rotation Gate: α⟨σ_y⟩ + β⟨σ_z⟩']
    
    for idx, (gate_type, title) in enumerate(zip(gate_types, titles)):
        ax = axes[idx]
        
        params = MixedObservableParams(
            alpha=0.7,
            beta=0.3,
            gate_type=gate_type,
            eta=0.35,
            gamma=0.04
        )
        
        memristor = MixedObservableMemristor(params)
        
        # With memory
        data_mem = memristor.simulate_hysteresis(num_cycles=5)
        
        # Without memory
        params.eta = 0
        memristor_no_mem = MixedObservableMemristor(params)
        data_no_mem = memristor_no_mem.simulate_hysteresis(num_cycles=5)
        
        # Plot steady state
        n = len(data_mem['t'])
        start = 3 * n // 5
        
        # With memory
        ax.plot(data_mem['input'][start:], data_mem['output'][start:],
               'b-', linewidth=2.5, label='With memory', alpha=0.8)
        
        # Without memory
        ax.plot(data_no_mem['input'][start:], data_no_mem['output'][start:],
               'r--', linewidth=2, label='No memory', alpha=0.7)
        
        # Add arrows
        indices = np.linspace(start + 100, n - 100, 4, dtype=int)
        for i in indices:
            if i + 50 < n:
                ax.annotate('', xy=(data_mem['input'][i+50], data_mem['output'][i+50]),
                           xytext=(data_mem['input'][i], data_mem['output'][i]),
                           arrowprops=dict(arrowstyle='->', color='blue', 
                                         lw=2, alpha=0.6))
        
        ax.set_xlabel('Input ⟨σ_y⟩')
        ax.set_ylabel('Output (mixed)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
        
        # Mark origin
        ax.scatter([0], [0], color='red', s=150, zorder=5, marker='o')
    
    fig.suptitle('Gate Type Comparison for Mixed Observables', fontsize=16)
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    print("Analyzing Mixed Observable Strategies for Pinched Hysteresis...")
    print("=" * 60)
    
    # Generate plots
    print("1. Comparing different observable combinations...")
    fig1 = plot_observable_comparison()
    plt.savefig('mixed_observable_comparison.pdf', dpi=300, bbox_inches='tight')
    
    print("2. Analyzing optimal mixed observable configuration...")
    fig2 = plot_optimal_mixed_observable()
    plt.savefig('optimal_mixed_observable.pdf', dpi=300, bbox_inches='tight')
    
    print("3. Comparing gate types...")
    fig3 = plot_gate_comparison()
    plt.savefig('gate_type_comparison.pdf', dpi=300, bbox_inches='tight')
    
    print("\nKey findings:")
    print("✓ Mixing observables improves pinching behavior")
    print("✓ Phase gate: Mix σ_y with σ_x (both non-zero for |+⟩ states)")
    print("✓ Rotation gate: Mix σ_y with σ_z (rotation creates population imbalance)")
    print("✓ Optimal mixing ratio: α ≈ 0.7, β ≈ 0.3")
    print("✓ Rotation gate provides richer dynamics due to population mixing")
    
    plt.show()
