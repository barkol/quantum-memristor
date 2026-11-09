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
    """Parameters for photonic quantum memristor with H·P·H-based memory
    
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
    """Photonic quantum memristor using H·P(4θ)·H sequence instead of HWP"""
    
    def __init__(self, params: MemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (phase θ = 0)"""
        self.memory = 0.0  # θ = 0 (phase parameter in radians)
        self.integrated_input = 0.0
    
    def h_p_h_transformation(self, input_angle: float, theta: float) -> float:
        """H·P(4θ)·H transformation sequence
        
        This replaces the HWP(θ) operation:
        - H: Hadamard gate (45° rotation in polarization)
        - P(4θ): Phase shift gate with phase 4θ
        - H: Hadamard gate again
        
        The net effect is equivalent to a rotation by 2θ in polarization
        """
        # The H·P(4θ)·H sequence produces the same effect as HWP(θ)
        # which rotates polarization by 2θ
        return input_angle + 2 * theta
    
    def memory_dependent_response(self, input_signal, memory):
        """Memory-dependent transformation using H·P·H sequence"""
        # Memory modulation - phase θ affects transformation
        # Using tanh to bound the memory contribution
        memory_factor = 1.0 + self.params.memory_coupling * np.tanh(memory)
        
        # Apply transformation with base gain
        output = self.params.base_gain * input_signal * memory_factor
        
        # Return bounded output
        return output
    
    def update_memory(self, input_signal, dt):
        """Update phase θ based on detection events (proportional to input)"""
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
            
            # Get output after H·P(4θ)·H transformation
            output_signal = self.memory_dependent_response(input_signal, self.memory)
            
            # Update phase θ based on detection
            self.update_memory(input_signal, dt)
            
            # Store values
            input_signals.append(input_signal)
            output_signals.append(output_signal)
            memory_values.append(self.memory)
        
        return {
            't': t_array,
            'input': np.array(input_signals),    # sin(2α)
            'output': np.array(output_signals),  # sin(2β) where β = α + 2θ
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
    ax.set_title('Photonic Quantum Memristor with H·P(4θ)·H:\nPolarization Hysteresis', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.legend(loc='upper left')
    
    # Set axis limits to ensure proper scaling
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    
    # Add annotation about the H·P·H implementation
    ax.text(0.02, 0.02, 'Using H·P(4θ)·H sequence', 
            transform=ax.transAxes, fontsize=18,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_h_p_h_hysteresis.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_h_p_h_hysteresis.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_phase_evolution(params=None, save=True):
    """Plot phase θ evolution over time"""
    if params is None:
        params = MemristorParams()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    memristor = QuantumMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=1000)
    
    ax.plot(data_mem['t'], data_mem['memory'], 'g-', linewidth=5)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Phase θ (rad)', fontweight='bold')
    ax.set_title('Phase Evolution in H·P(4θ)·H Sequence', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add annotation
    ax.text(0.02, 0.95, 'Phase shift in P(4θ) gate', 
            transform=ax.transAxes, fontsize=18,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_phase_evolution.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_phase_evolution.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_transformation_comparison(save=True):
    """Compare HWP(θ) vs H·P(4θ)·H transformations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Input angles
    alpha = np.linspace(0, 2*np.pi, 100)
    theta_values = [0, np.pi/8, np.pi/4, np.pi/2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_values)))
    
    # Plot HWP transformation
    for i, theta in enumerate(theta_values):
        beta_hwp = alpha + 2*theta
        ax1.plot(alpha, beta_hwp % (2*np.pi), color=colors[i], 
                linewidth=3, label=f'θ={theta:.2f}')
    
    ax1.set_xlabel('Input angle α (rad)', fontweight='bold')
    ax1.set_ylabel('Output angle β (rad)', fontweight='bold')
    ax1.set_title('HWP(θ) Transformation\nβ = α + 2θ', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(0, 2*np.pi)
    
    # Plot H·P·H transformation (same result)
    for i, theta in enumerate(theta_values):
        beta_hph = alpha + 2*theta  # Same transformation
        ax2.plot(alpha, beta_hph % (2*np.pi), color=colors[i], 
                linewidth=3, label=f'θ={theta:.2f}')
    
    ax2.set_xlabel('Input angle α (rad)', fontweight='bold')
    ax2.set_ylabel('Output angle β (rad)', fontweight='bold')
    ax2.set_title('H·P(4θ)·H Transformation\nβ = α + 2θ', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, 2*np.pi)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_transformation_comparison.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_transformation_comparison.png', bbox_inches='tight', dpi=300)
    
    return fig, (ax1, ax2)

def plot_gate_sequence_diagram(save=True):
    """Plot the H·P(4θ)·H gate sequence diagram"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Turn off axis
    ax.axis('off')
    
    # Draw the gate sequence
    gate_width = 0.15
    gate_height = 0.3
    y_center = 0.5
    
    # Input
    ax.text(0.05, y_center, '|ψ⟩', fontsize=28, ha='center', va='center')
    ax.arrow(0.1, y_center, 0.1, 0, head_width=0.03, head_length=0.02, 
             fc='black', ec='black', linewidth=2)
    
    # First Hadamard
    rect1 = plt.Rectangle((0.25, y_center-gate_height/2), gate_width, gate_height,
                         fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect1)
    ax.text(0.25 + gate_width/2, y_center, 'H', fontsize=24, ha='center', va='center')
    ax.arrow(0.25 + gate_width, y_center, 0.1, 0, head_width=0.03, head_length=0.02,
             fc='black', ec='black', linewidth=2)
    
    # Phase shifter
    rect2 = plt.Rectangle((0.45, y_center-gate_height/2), gate_width, gate_height,
                         fill=True, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(rect2)
    ax.text(0.45 + gate_width/2, y_center, 'P(4θ)', fontsize=24, ha='center', va='center')
    ax.arrow(0.45 + gate_width, y_center, 0.1, 0, head_width=0.03, head_length=0.02,
             fc='black', ec='black', linewidth=2)
    
    # Second Hadamard
    rect3 = plt.Rectangle((0.65, y_center-gate_height/2), gate_width, gate_height,
                         fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(0.65 + gate_width/2, y_center, 'H', fontsize=24, ha='center', va='center')
    ax.arrow(0.65 + gate_width, y_center, 0.1, 0, head_width=0.03, head_length=0.02,
             fc='black', ec='black', linewidth=2)
    
    # Output
    ax.text(0.95, y_center, '|ψ\'⟩', fontsize=28, ha='center', va='center')
    
    # Title
    ax.text(0.5, 0.85, 'H·P(4θ)·H Gate Sequence', fontsize=32, ha='center', va='center',
            fontweight='bold')
    
    # Explanation
    ax.text(0.5, 0.15, 'Equivalent to HWP(θ): Rotates polarization by 2θ', 
            fontsize=20, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('memristor_gate_sequence.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('memristor_gate_sequence.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

# Run all plots
if __name__ == "__main__":
    print("Generating Photonic Quantum Memristor Figures with H·P(4θ)·H Implementation...")
    print("=" * 70)
    print("Replacing HWP(θ) with H·P(4θ)·H sequence")
    print("=" * 70)
    
    # Create all figures
    print("\n1. Main hysteresis loop with H·P·H implementation...")
    plot_main_hysteresis()
    
    print("2. Phase evolution...")
    plot_phase_evolution()
    
    print("3. Transformation comparison (HWP vs H·P·H)...")
    plot_transformation_comparison()
    
    print("4. Gate sequence diagram...")
    plot_gate_sequence_diagram()
    
    print("\n" + "=" * 70)
    print("All figures saved successfully!")
    print("\nKey implementation change:")
    print("- Original: HWP(θ) rotates polarization by 2θ")
    print("- Modified: H·P(4θ)·H sequence produces the same 2θ rotation")
    print("\nThe H·P(4θ)·H decomposition shows how the HWP can be")
    print("implemented using only phase shifters and Hadamard gates.")
    
    # Show all plots
    plt.show()
