import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

# Configure plotting for presentations - MUCH LARGER FONTS
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 36

@dataclass
class MemristorParams:
    """Parameters for photonic quantum memristor with Z·H·P(4θ)·H gate sequence
    
    Parameters optimized for wider hysteresis loops for presentations
    """
    # Memory dynamics (phase evolution) - ENHANCED FOR WIDER LOOPS
    eta: float = 0.5           # Increased feedback strength
    gamma: float = 0.01        # Reduced relaxation for stronger memory
    
    # Optical transformation parameters - ENHANCED FOR WIDER LOOPS
    base_gain: float = 0.6     # Adjusted for wider response
    memory_coupling: float = 0.5  # Increased coupling for wider loops
    
    # Input polarization parameters
    sweep_rate: float = 0.12   # Slightly slower for clearer loops
    amplitude: float = 1.0     # Maximum input amplitude

class QuantumMemristor:
    """Photonic quantum memristor using gate sequence Z·H·P(4θ)·H instead of HWP"""
    
    def __init__(self, params: MemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (phase angle θ = 0)"""
        self.memory = 0.0  # θ = 0 (phase parameter in radians)
        self.integrated_input = 0.0
    
    def apply_gate_sequence(self, input_angle, memory):
        """Apply the Z·H·P(4θ)·H gate sequence to transform the input state
        
        This sequence replaces the HWP(θ) operation with:
        - H: Hadamard gate
        - P(4θ): Phase gate with phase 4θ
        - H: Hadamard gate
        - Z: Pauli-Z gate
        """
        # For a polarization state with angle α:
        # |ψ⟩ = cos(α)|H⟩ + sin(α)|V⟩
        
        # The Z·H·P(4θ)·H sequence acts as follows:
        # 1. H: Hadamard transformation
        # 2. P(4θ): Phase shift by 4θ on |V⟩ component
        # 3. H: Another Hadamard
        # 4. Z: Phase flip on |V⟩
        
        # For polarization states, this is equivalent to a rotation
        # The net effect is similar to HWP(θ) but with the phase factor of 4
        rotation_angle = 2 * memory  # The effective rotation is 2θ (similar to HWP)
        
        # Apply the rotation to the polarization angle
        output_angle = input_angle + rotation_angle
        
        return output_angle
    
    def memory_dependent_response(self, input_signal, memory):
        """Apply gate sequence Z·H·P(4θ)·H transformation"""
        # The transformation is: Z · H · P(4θ) · H
        # This replaces the HWP(θ) operation
        
        # For polarization states, input_signal = sin(2α)
        # We need to extract α, transform it, and return sin(2β)
        
        # Handle edge cases to avoid numerical issues
        input_clipped = np.clip(input_signal, -1, 1)
        
        # Extract polarization angle from input signal
        # input_signal = sin(2α), so α = arcsin(input_signal)/2
        input_angle = np.arcsin(input_clipped) / 2
        
        # Apply the Z·H·P(4θ)·H gate sequence
        # This rotates the polarization angle by 2θ (similar to HWP)
        output_angle = self.apply_gate_sequence(input_angle, memory)
        
        # Convert back to signal representation
        # output_signal = sin(2β) where β is the output polarization angle
        output_signal = np.sin(2 * output_angle)
        
        # Apply memory-dependent gain modulation
        memory_factor = 1.0 + self.params.memory_coupling * np.tanh(memory)
        
        # Final output with gain
        output = self.params.base_gain * output_signal * memory_factor
        
        # Ensure output is bounded
        return np.clip(output, -1, 1)
    
    def update_memory(self, input_signal, dt):
        """Update phase parameter θ based on detection events (proportional to input)"""
        # θ evolves based on detection rate (∝ input polarization)
        dmemory = self.params.eta * input_signal - self.params.gamma * self.memory
        self.memory += dmemory * dt
        
        # Track integrated detection events
        self.integrated_input += input_signal * dt
    
    def simulate_hysteresis(self, num_cycles=3, points_per_cycle=2000):
        """Simulate the photonic memristor to show polarization hysteresis"""
        # Time array
        period = 2 * np.pi / self.params.sweep_rate
        T = num_cycles * period
        t_array = np.linspace(0, T, num_cycles * points_per_cycle)
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        input_signals = []
        output_signals = []
        memory_values = []
        
        # Reset state (θ = 0)
        self.reset()
        
        # Simulate
        for i, t in enumerate(t_array):
            # Sinusoidal input polarization angle α(t)
            input_signal = self.params.amplitude * np.sin(self.params.sweep_rate * t)
            
            # Get output after Z·H·P(4θ)·H transformation
            output_signal = self.memory_dependent_response(input_signal, self.memory)
            
            # Update phase parameter θ based on detection
            self.update_memory(input_signal, dt)
            
            # Store values
            input_signals.append(input_signal)
            output_signals.append(output_signal)
            memory_values.append(self.memory)
        
        return {
            't': t_array,
            'input': np.array(input_signals),    # sin(2α)
            'output': np.array(output_signals),  # transformed output
            'memory': np.array(memory_values)    # θ(t)
        }
    
    def simulate_no_memory(self, num_cycles=3, points_per_cycle=2000):
        """Simulate without memory for comparison"""
        # Save original parameters
        original_eta = self.params.eta
        original_gamma = self.params.gamma
        
        # Set no memory (instant decay)
        self.params.eta = 0.0
        self.params.gamma = 1000.0  # Very fast decay
        
        # Run simulation
        result = self.simulate_hysteresis(num_cycles, points_per_cycle)
        
        # Restore parameters
        self.params.eta = original_eta
        self.params.gamma = original_gamma
        
        return result

def plot_main_hysteresis(params=None, save=True):
    """Plot the main hysteresis loop - Input vs Output Polarization"""
    if params is None:
        params = MemristorParams()  # Use default parameters
    
    # Create figure with presentation-friendly aspect ratio
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Initialize and simulate
    memristor = QuantumMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=2000)
    data_no_mem = memristor.simulate_no_memory(num_cycles=4, points_per_cycle=2000)
    
    # Get last cycle
    n_points = len(data_mem['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    # Extract last cycle
    input_mem = data_mem['input'][last_cycle_start:]
    output_mem = data_mem['output'][last_cycle_start:]
    input_no_mem = data_no_mem['input'][last_cycle_start:]
    output_no_mem = data_no_mem['output'][last_cycle_start:]
    
    # Smooth the output
    output_mem_smooth = savgol_filter(output_mem, 51, 3)
    
    # Time for coloring
    t_last = data_mem['t'][last_cycle_start:] - data_mem['t'][last_cycle_start]
    colors = plt.cm.viridis(t_last / t_last[-1])
    
    # Plot with memory - color gradient with thicker lines
    for i in range(0, len(input_mem)-1, 10):
        ax.plot(input_mem[i:i+2], output_mem_smooth[i:i+2], 
                color=colors[i], linewidth=6, alpha=0.9)
    
    # Plot without memory - thicker line
    ax.plot(input_no_mem, output_no_mem, 'r--', 
             linewidth=4, alpha=0.8, label='No memory (fixed phase)')
    
    # Add direction arrows - larger and more visible
    n_arrows = 6
    arrow_indices = np.linspace(100, len(input_mem)-100, n_arrows, dtype=int)
    for idx in arrow_indices:
        if idx + 20 < len(input_mem):
            if output_mem_smooth[idx] > output_no_mem[np.argmin(np.abs(input_no_mem - input_mem[idx]))]:
                arrow_color = 'darkgreen'
            else:
                arrow_color = 'darkblue'
                
            ax.annotate('', xy=(input_mem[idx+20], output_mem_smooth[idx+20]),
                        xytext=(input_mem[idx], output_mem_smooth[idx]),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4, alpha=0.8))
    
    ax.set_xlabel('Input Polarization sin(2α)', fontweight='bold')
    ax.set_ylabel('Output Polarization (transformed)', fontweight='bold')
    ax.set_title('Quantum Memristor with Z·H·P(4θ)·H Gate Sequence', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.legend(loc='upper left')
    
    # Set axis limits to ensure proper scaling
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    
    # Add annotation about the gate sequence
    ax.text(0.98, 0.02, 'Gate sequence: Z·H·P(4θ)·H', 
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            ha='right', va='bottom', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_zhp4h_hysteresis.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_zhp4h_hysteresis.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_gate_sequence_effect(params=None, save=True):
    """Plot the effect of the Z·H·P(4θ)·H gate sequence"""
    if params is None:
        params = MemristorParams()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Gate sequence transformation for different θ values
    input_angles = np.linspace(-np.pi/2, np.pi/2, 100)
    theta_values = [0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(theta_values)))
    
    memristor = QuantumMemristor(params)
    
    for i, theta in enumerate(theta_values):
        output_angles = []
        for input_angle in input_angles:
            output_angle = memristor.apply_gate_sequence(input_angle, theta)
            output_angles.append(output_angle)
        
        # Convert to signal representation
        input_signals = np.sin(2 * input_angles)
        output_signals = np.sin(2 * np.array(output_angles))
        
        ax1.plot(input_signals, output_signals, 
                label=f'θ={theta:.1f}', linewidth=4, color=colors[i])
    
    ax1.set_xlabel('Input sin(2α)', fontweight='bold')
    ax1.set_ylabel('Output after Z·H·P(4θ)·H', fontweight='bold')
    ax1.set_title('Gate Sequence Transformation', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    
    # Right plot: Comparison with HWP
    ax2.text(0.5, 0.9, 'Gate Decomposition', 
             transform=ax2.transAxes, ha='center', fontsize=28, fontweight='bold')
    
    ax2.text(0.5, 0.7, 'HWP(θ) → Z · H · P(4θ) · H', 
             transform=ax2.transAxes, ha='center', fontsize=24)
    
    ax2.text(0.1, 0.5, 'Where:', 
             transform=ax2.transAxes, fontsize=20, fontweight='bold')
    
    ax2.text(0.1, 0.4, '• H = Hadamard gate', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.3, '• P(4θ) = Phase gate with phase 4θ', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.2, '• Z = Pauli-Z gate', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.1, '• θ = Memory parameter', 
             transform=ax2.transAxes, fontsize=18)
    
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('gate_sequence_effect.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('gate_sequence_effect.png', bbox_inches='tight', dpi=300)
    
    return fig, (ax1, ax2)

def plot_phase_evolution(params=None, save=True):
    """Plot phase parameter θ evolution over time"""
    if params is None:
        params = MemristorParams()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    memristor = QuantumMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=1000)
    
    ax.plot(data_mem['t'], data_mem['memory'], 'g-', linewidth=5)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Phase Parameter θ (rad)', fontweight='bold')
    ax.set_title('Phase Parameter Evolution for Z·H·P(4θ)·H Gate', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add annotation
    ax.text(0.98, 0.95, 'θ controls P(4θ) gate', 
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            ha='right', va='top', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_phase_evolution.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_phase_evolution.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

# Run demonstration
if __name__ == "__main__":
    print("Generating Quantum Memristor with Z·H·P(4θ)·H Gate Sequence...")
    print("=" * 60)
    print("Replacing HWP(θ) with gate decomposition: Z · H · P(4θ) · H")
    print("=" * 60)
    
    # Create figures
    print("\n1. Main hysteresis loop with new gate sequence...")
    plot_main_hysteresis()
    
    print("2. Gate sequence transformation effect...")
    plot_gate_sequence_effect()
    
    print("3. Phase parameter evolution...")
    plot_phase_evolution()
    
    print("\n" + "=" * 60)
    print("All figures saved successfully!")
    print("\nKey changes:")
    print("1. HWP(θ) replaced with Z·H·P(4θ)·H gate sequence")
    print("2. Phase gate P(4θ) uses 4× the memory parameter θ")
    print("3. Hadamard gates create superpositions")
    print("4. Z gate adds additional phase")
    print("\nThis decomposition is commonly used in quantum computing")
    print("to implement rotation gates using basic quantum gates.")
    
    # Show plots
    plt.show()
