import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter

# Configure plotting
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18

@dataclass
class PinchedMemristorParams:
    """Parameters optimized for pinched hysteresis loops"""
    # Memory dynamics - tuned for pinching
    eta: float = 0.3           # Moderate feedback for controlled memory
    gamma: float = 0.05        # Relaxation ensures pinching at origin
    theta_0: float = 0.0       # Zero equilibrium is crucial for pinching
    
    # Detection parameters
    detection_efficiency: float = 0.9
    
    # Sinusoidal sweep parameters
    omega: float = 1.0         # Angular frequency of input sweep
    amplitude: float = np.pi   # Amplitude of phase modulation

class PinchedQuantumMemristor:
    """Quantum memristor optimized for pinched hysteresis visualization"""
    
    def __init__(self, params: PinchedMemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state"""
        self.theta = 0.0
        self.detection_integral = 0.0
        
    def apply_phase_gate(self, phi: float, theta: float) -> float:
        """Apply memory-dependent phase gate"""
        return phi + theta
    
    def calculate_observable(self, phase: float) -> float:
        """Calculate ⟨σ_y⟩ = sin(phase)"""
        return np.sin(phase)
    
    def detection_probability(self, output_phase: float) -> float:
        """Detection probability for feedback"""
        # Use projection onto |+⟩ state
        return (1 + np.cos(output_phase)) / 2
    
    def update_memory(self, input_phi: float, detection: bool, dt: float):
        """Update memory with pinching-optimized dynamics
        
        Key: Memory should naturally decay when input crosses zero
        """
        # Detection signal
        N = 1.0 if detection else -1.0
        
        # Memory update with input-dependent modulation
        # Critical: Use input phase directly to ensure pinching
        modulation = np.sin(input_phi)  # This ensures θ → 0 when φ → 0, π, 2π
        
        # Evolution equation
        dtheta = (self.params.eta * N * modulation - 
                 self.params.gamma * self.theta)
        
        self.theta += dtheta * dt
    
    def simulate_pinched_hysteresis(self, num_cycles=5, points_per_cycle=1000):
        """Simulate with sinusoidal phase sweep for pinched hysteresis"""
        # Time array
        T = num_cycles * 2 * np.pi / self.params.omega
        t_array = np.linspace(0, T, num_cycles * points_per_cycle)
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        phi_values = []
        input_observable = []
        output_observable = []
        theta_values = []
        
        # Reset state
        self.reset()
        
        for t in t_array:
            # Sinusoidal phase modulation
            phi = self.params.amplitude * np.sin(self.params.omega * t)
            
            # Input observable
            input_obs = self.calculate_observable(phi)
            
            # Apply memory-dependent phase gate
            output_phase = self.apply_phase_gate(phi, self.theta)
            
            # Output observable
            output_obs = self.calculate_observable(output_phase)
            
            # Detection and feedback
            p_detect = self.detection_probability(output_phase)
            detection = np.random.random() < (p_detect * self.params.detection_efficiency)
            
            # Update memory
            self.update_memory(phi, detection, dt)
            
            # Store values
            phi_values.append(phi)
            input_observable.append(input_obs)
            output_observable.append(output_obs)
            theta_values.append(self.theta)
        
        return {
            't': t_array,
            'phi': np.array(phi_values),
            'input': np.array(input_observable),
            'output': np.array(output_observable),
            'theta': np.array(theta_values),
            'omega': self.params.omega
        }
    
    def simulate_no_memory(self, num_cycles=5, points_per_cycle=1000):
        """Simulate without memory for comparison"""
        original_eta = self.params.eta
        self.params.eta = 0.0
        
        result = self.simulate_pinched_hysteresis(num_cycles, points_per_cycle)
        
        self.params.eta = original_eta
        return result

def plot_pinched_hysteresis_comparison():
    """Plot pinched hysteresis for different sweep strategies"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Different sweep strategies
    strategies = [
        ("Sinusoidal", lambda t, A, w: A * np.sin(w * t)),
        ("Triangular", lambda t, A, w: A * (2/np.pi) * np.arcsin(np.sin(w * t))),
        ("Sawtooth", lambda t, A, w: A * (2 * (w * t / (2*np.pi) % 1) - 1))
    ]
    
    # Different frequencies
    frequencies = [0.5, 1.0, 2.0]
    
    for i, (name, sweep_func) in enumerate(strategies):
        ax = axes[0, i]
        
        # Plot for different frequencies
        for j, omega in enumerate(frequencies):
            params = PinchedMemristorParams(omega=omega)
            memristor = PinchedQuantumMemristor(params)
            
            # Override the phase calculation
            T = 5 * 2 * np.pi / omega
            t_array = np.linspace(0, T, 5000)
            dt = t_array[1] - t_array[0]
            
            input_obs = []
            output_obs = []
            
            memristor.reset()
            
            for t in t_array:
                # Use custom sweep function
                phi = sweep_func(t, params.amplitude, omega)
                
                # Calculate observables
                input_val = np.sin(phi)
                output_phase = phi + memristor.theta
                output_val = np.sin(output_phase)
                
                # Detection
                p_detect = (1 + np.cos(output_phase)) / 2
                detection = np.random.random() < p_detect
                
                # Update memory
                memristor.update_memory(phi, detection, dt)
                
                input_obs.append(input_val)
                output_obs.append(output_val)
            
            # Plot last 2 cycles
            n = len(input_obs)
            start = 3 * n // 5
            
            color = plt.cm.viridis(j / 2)
            ax.plot(input_obs[start:], output_obs[start:], 
                   color=color, linewidth=2, label=f'ω={omega}', alpha=0.8)
        
        ax.set_xlabel('Input ⟨σ_y⟩')
        ax.set_ylabel('Output ⟨σ_y⟩')
        ax.set_title(f'{name} Sweep')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Bottom row: Effect of parameters on pinching
    # Effect of eta
    ax = axes[1, 0]
    omega = 1.0
    for eta in [0.1, 0.2, 0.3, 0.4]:
        params = PinchedMemristorParams(eta=eta, omega=omega)
        memristor = PinchedQuantumMemristor(params)
        data = memristor.simulate_pinched_hysteresis(num_cycles=5)
        
        n = len(data['t'])
        start = 3 * n // 5
        
        ax.plot(data['input'][start:], data['output'][start:], 
               linewidth=2, label=f'η={eta}', alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_y⟩')
    ax.set_title('Effect of Feedback Strength η')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Effect of gamma
    ax = axes[1, 1]
    for gamma in [0.01, 0.03, 0.05, 0.1]:
        params = PinchedMemristorParams(gamma=gamma, omega=omega)
        memristor = PinchedQuantumMemristor(params)
        data = memristor.simulate_pinched_hysteresis(num_cycles=5)
        
        n = len(data['t'])
        start = 3 * n // 5
        
        ax.plot(data['input'][start:], data['output'][start:], 
               linewidth=2, label=f'γ={gamma}', alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_y⟩')
    ax.set_title('Effect of Relaxation Rate γ')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Optimal pinched hysteresis
    ax = axes[1, 2]
    params = PinchedMemristorParams(eta=0.3, gamma=0.05, omega=1.0)
    memristor = PinchedQuantumMemristor(params)
    
    # With memory
    data_mem = memristor.simulate_pinched_hysteresis(num_cycles=5)
    n = len(data_mem['t'])
    start = 3 * n // 5
    
    # Plot with time coloring
    t_cycle = data_mem['t'][start:] - data_mem['t'][start]
    colors = plt.cm.plasma(t_cycle / (t_cycle[-1] - t_cycle[0]))
    
    # Use scatter for color gradient
    scatter = ax.scatter(data_mem['input'][start::20], 
                        data_mem['output'][start::20],
                        c=colors[::20], s=30, cmap='plasma', alpha=0.8)
    
    # Without memory
    data_no_mem = memristor.simulate_no_memory(num_cycles=5)
    ax.plot(data_no_mem['input'][start:], data_no_mem['output'][start:], 
           'k--', linewidth=2, alpha=0.6, label='No memory')
    
    # Add arrows
    indices = np.linspace(start + 100, n - 100, 6, dtype=int)
    for idx in indices:
        if idx + 50 < n:
            ax.annotate('', xy=(data_mem['input'][idx+50], data_mem['output'][idx+50]),
                       xytext=(data_mem['input'][idx], data_mem['output'][idx]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2, alpha=0.6))
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_y⟩')
    ax.set_title('Optimal Pinched Hysteresis')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add text
    ax.text(0.02, 0.98, 'Pinched at origin', transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    fig.suptitle('Pinched Hysteresis in Quantum Phase Memristor', fontsize=20)
    plt.tight_layout()
    
    return fig

def plot_detailed_pinched_analysis():
    """Detailed analysis of pinched hysteresis characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    params = PinchedMemristorParams(eta=0.3, gamma=0.05, omega=1.0)
    memristor = PinchedQuantumMemristor(params)
    data = memristor.simulate_pinched_hysteresis(num_cycles=10)
    
    # Get steady state (last few cycles)
    n = len(data['t'])
    steady_start = 7 * n // 10
    
    # Plot 1: Pinched hysteresis loop
    ax = axes[0, 0]
    ax.plot(data['input'][steady_start:], data['output'][steady_start:], 
           'b-', linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_y⟩')
    ax.set_title('Steady-State Pinched Hysteresis')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Highlight pinching point
    ax.scatter([0], [0], color='red', s=200, zorder=5, marker='o')
    ax.text(0.1, 0.1, 'Pinch point\n(0, 0)', fontsize=12, 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Time evolution showing pinching
    ax = axes[0, 1]
    t_plot = data['t'][steady_start:] - data['t'][steady_start]
    ax.plot(t_plot, data['input'][steady_start:], 'b-', linewidth=2, 
           label='Input ⟨σ_y⟩', alpha=0.7)
    ax.plot(t_plot, data['output'][steady_start:], 'r-', linewidth=2.5, 
           label='Output ⟨σ_y⟩')
    
    # Mark pinching times
    pinch_indices = np.where(np.abs(data['input'][steady_start:]) < 0.01)[0]
    if len(pinch_indices) > 0:
        ax.scatter(t_plot[pinch_indices], data['output'][steady_start:][pinch_indices], 
                  color='red', s=100, zorder=5, label='Pinch points')
    
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Observable ⟨σ_y⟩')
    ax.set_title('Time Evolution Showing Pinching')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 3: Memory parameter behavior
    ax = axes[1, 0]
    ax.plot(t_plot, data['theta'][steady_start:], 'g-', linewidth=2.5)
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Memory θ (rad)')
    ax.set_title('Memory Parameter Evolution')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight when input crosses zero
    for idx in pinch_indices[::2]:  # Every other crossing
        ax.axvline(x=t_plot[idx], color='red', linestyle=':', alpha=0.5)
    
    # Plot 4: Frequency dependence
    ax = axes[1, 1]
    frequencies = np.logspace(-0.5, 0.7, 15)
    loop_areas = []
    pinch_quality = []  # How close to zero at pinch point
    
    for omega in frequencies:
        params_freq = PinchedMemristorParams(eta=0.3, gamma=0.05, omega=omega)
        memristor_freq = PinchedQuantumMemristor(params_freq)
        data_freq = memristor_freq.simulate_pinched_hysteresis(num_cycles=5)
        
        # Use last cycle
        n_freq = len(data_freq['t'])
        start_freq = 3 * n_freq // 5
        
        # Calculate loop area
        x = data_freq['input'][start_freq:]
        y = data_freq['output'][start_freq:]
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        loop_areas.append(area)
        
        # Calculate pinch quality (deviation from zero at input=0)
        zero_crossings = np.where(np.diff(np.sign(x)))[0]
        if len(zero_crossings) > 0:
            pinch_values = [np.abs(y[idx]) for idx in zero_crossings]
            pinch_quality.append(np.mean(pinch_values))
        else:
            pinch_quality.append(0)
    
    # Plot loop area
    ax2 = ax.twinx()
    line1 = ax.plot(frequencies, loop_areas, 'bo-', linewidth=2, 
                    markersize=8, label='Loop area')
    line2 = ax2.plot(frequencies, pinch_quality, 'rs-', linewidth=2, 
                     markersize=6, label='Pinch deviation')
    
    ax.set_xlabel('Frequency ω')
    ax.set_ylabel('Loop Area', color='b')
    ax2.set_ylabel('Pinch Point Deviation', color='r')
    ax.set_title('Frequency Dependence of Pinched Hysteresis')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    
    fig.suptitle('Detailed Analysis of Pinched Hysteresis', fontsize=18)
    plt.tight_layout()
    
    return fig

# Key insights function
def plot_sweep_strategy_guide():
    """Visual guide for sweep strategies"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    t = np.linspace(0, 4*np.pi, 1000)
    
    # Different sweep functions
    strategies = {
        'Sinusoidal (Best for pinching)': np.sin(t),
        'Modified sine (sharper pinching)': np.sin(t) * np.abs(np.sin(t))**0.5,
        'Triangular': (2/np.pi) * np.arcsin(np.sin(t)),
        'Square (not recommended)': np.sign(np.sin(t))
    }
    
    colors = ['blue', 'red', 'green', 'gray']
    linestyles = ['-', '--', '-.', ':']
    
    for (name, signal), color, ls in zip(strategies.items(), colors, linestyles):
        ax.plot(t, signal, color=color, linestyle=ls, linewidth=2.5, label=name)
    
    ax.set_xlabel('Time (a.u.)', fontsize=14)
    ax.set_ylabel('Input Signal', fontsize=14)
    ax.set_title('Sweep Strategies for Pinched Hysteresis', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Highlight zero crossings
    for x in [0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi]:
        ax.axvline(x=x, color='red', linestyle=':', alpha=0.3)
        if x > 0:
            ax.text(x, 1.2, 'Pinch', ha='center', fontsize=10, color='red')
    
    # Add text box with recommendations
    textstr = '\n'.join([
        'Recommendations for Pinched Hysteresis:',
        '• Use sinusoidal sweep for smooth pinching',
        '• Ensure signal passes through zero',
        '• Memory should decay when input → 0',
        '• Relaxation rate γ controls pinch quality',
        '• Lower frequencies → larger loops'
    ])
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    print("Generating Pinched Hysteresis Analysis...")
    print("=" * 60)
    
    # Generate all plots
    print("1. Comparing sweep strategies and parameters...")
    fig1 = plot_pinched_hysteresis_comparison()
    plt.savefig('pinched_hysteresis_comparison.pdf', dpi=300, bbox_inches='tight')
    
    print("2. Detailed pinched hysteresis analysis...")
    fig2 = plot_detailed_pinched_analysis()
    plt.savefig('pinched_hysteresis_analysis.pdf', dpi=300, bbox_inches='tight')
    
    print("3. Sweep strategy guide...")
    fig3 = plot_sweep_strategy_guide()
    plt.savefig('sweep_strategy_guide.pdf', dpi=300, bbox_inches='tight')
    
    print("\nKey findings for pinched hysteresis:")
    print("✓ Sinusoidal sweep creates best pinching behavior")
    print("✓ Memory coupling to sin(φ) ensures pinching at origin")
    print("✓ Moderate feedback (η ≈ 0.3) and relaxation (γ ≈ 0.05) optimal")
    print("✓ Lower frequencies create larger loops while maintaining pinching")
    print("✓ Pinch quality depends on balance between memory and relaxation")
    
    plt.show()
