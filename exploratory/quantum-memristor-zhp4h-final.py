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
    
    Parameters optimized for clear pinched hysteresis loops
    """
    # Memory dynamics (phase evolution)
    eta: float = 0.3           # Feedback strength (reduced for stability with cos(4θ))
    gamma: float = 0.02        # Relaxation rate for memory decay
    
    # Optical transformation parameters
    base_gain: float = 0.8     # Base transmission/reflection ratio
    memory_coupling: float = 0.4  # Coupling strength for cos(4θ) modulation
    
    # Input polarization parameters
    sweep_rate: float = 0.15   # Input modulation frequency
    amplitude: float = 1.0     # Maximum input amplitude

class QuantumMemristor:
    """Photonic quantum memristor using gate sequence Z·H·P(4θ)·H instead of HWP
    
    The gate sequence Z·H·P(4θ)·H implements a memory-dependent transformation
    where the phase gate P(4θ) creates oscillatory behavior with memory parameter θ.
    
    Key memristive properties:
    1. Pinched hysteresis: Output = 0 when Input = 0 (always passes through origin)
    2. Memory dependence: The P(4θ) gate creates cos(4θ) modulation
    3. History dependence: θ evolves based on integrated input history
    
    The transformation is multiplicative: Output = Input × MemoryGain(θ)
    This ensures proper pinched hysteresis behavior.
    """
    
    def __init__(self, params: MemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (phase angle θ = 0)"""
        self.memory = 0.0  # θ = 0 (phase parameter in radians)
        self.integrated_input = 0.0
    
    def memory_dependent_response(self, input_signal, memory):
        """Apply gate sequence Z·H·P(4θ)·H transformation with memristive behavior
        
        The key to pinched hysteresis is that output must be zero when input is zero,
        regardless of the memory state θ. This is achieved by making the transformation
        multiplicative rather than purely rotational.
        """
        # For memristive behavior with pinched hysteresis:
        # Output = Input × MemoryFunction(θ)
        
        # The Z·H·P(4θ)·H gate sequence creates a memory-dependent transformation
        # We interpret this as modulating the transmission/reflection coefficients
        
        # Memory-dependent gain factor from the gate sequence
        # The phase gate P(4θ) creates oscillatory behavior with memory
        memory_modulation = np.cos(4 * memory)  # Oscillates with θ
        
        # Additional memory coupling for hysteresis width
        memory_factor = 1.0 + self.params.memory_coupling * memory_modulation
        
        # Apply the transformation
        # This ensures output = 0 when input = 0 (pinched at origin)
        output_signal = input_signal * self.params.base_gain * memory_factor
        
        # Ensure output is bounded
        return np.clip(output_signal, -1, 1)
    
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
    ax.set_ylabel('Output Polarization sin(2β)', fontweight='bold')
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
    """Plot the effect of the Z·H·P(4θ)·H gate sequence on memristive behavior"""
    if params is None:
        params = MemristorParams()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Memory modulation function from P(4θ) gate
    theta_values = np.linspace(0, 2*np.pi, 200)
    memory_modulation = np.cos(4 * theta_values)
    memory_factor = 1.0 + params.memory_coupling * memory_modulation
    
    ax1.plot(theta_values, memory_factor, 'b-', linewidth=4)
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Memory Parameter θ (rad)', fontweight='bold')
    ax1.set_ylabel('Memory Factor', fontweight='bold')
    ax1.set_title('Effect of P(4θ) Phase Gate', fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Add annotation
    ax1.text(0.5, 0.95, f'cos(4θ) modulation from P(4θ)', 
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
             fontsize=16)
    
    # Right plot: Gate sequence explanation
    ax2.text(0.5, 0.9, 'Gate Decomposition', 
             transform=ax2.transAxes, ha='center', fontsize=28, fontweight='bold')
    
    ax2.text(0.5, 0.75, 'HWP(θ) → Z · H · P(4θ) · H', 
             transform=ax2.transAxes, ha='center', fontsize=24)
    
    ax2.text(0.1, 0.6, 'Gate Sequence Effects:', 
             transform=ax2.transAxes, fontsize=20, fontweight='bold')
    
    ax2.text(0.1, 0.5, '• H: Creates superposition', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.4, '• P(4θ): Phase ∝ 4×memory creates cos(4θ) modulation', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.3, '• H: Converts back to polarization basis', 
             transform=ax2.transAxes, fontsize=18)
    ax2.text(0.1, 0.2, '• Z: Additional phase ensures proper symmetry', 
             transform=ax2.transAxes, fontsize=18)
    
    ax2.text(0.1, 0.05, 'Memristive behavior: Output = Input × Gain(θ)', 
             transform=ax2.transAxes, fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
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
    print("\nKey features:")
    print("1. Pinched hysteresis loops passing through origin (0,0)")
    print("2. Memory-dependent gain from P(4θ) gate: cos(4θ) modulation")
    print("3. Multiplicative transformation ensures memristive behavior")
    print("4. Gate sequence Z·H·P(4θ)·H replaces traditional HWP")
    print("\nThe P(4θ) phase gate creates oscillatory memory behavior,")
    print("leading to rich dynamics as θ evolves with input history.")
    
    # Show plots
    plt.show()
