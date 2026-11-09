import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

# Configure plotting for presentations
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 36

@dataclass
class PhaseShiftMemristorParams:
    """Parameters for quantum memristor with phase shift gate
    
    The memory parameter θ represents the phase shift applied by the gate
    """
    # Memory dynamics (phase shift evolution)
    eta: float = 0.4           # Feedback strength
    gamma: float = 0.02        # Relaxation rate for memory persistence
    
    # Memory coupling parameters
    memory_coupling: float = 0.8  # How strongly memory affects transformation
    
    # Initial phase offset
    phi_initial: float = 0.025  # Starting phase for input state preparation

class PhaseShiftQuantumMemristor:
    """Quantum memristor using phase shift gate with memory-dependent phase θ"""
    
    def __init__(self, params: PhaseShiftMemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (θ = 0)"""
        self.memory = 0.0  # θ = 0 (phase shift in radians)
        self.integrated_input = 0.0
    
    def calculate_sigma_y(self, phi, theta=0):
        """Calculate ⟨σ_y⟩ for state |ψ⟩ = 1/√2(|0⟩ + e^(i(φ/2 + θ))|1⟩)
        
        For this state: ⟨σ_y⟩ = -sin(φ/2 + θ)
        """
        return -np.sin(phi/2 + theta)
    
    def memory_dependent_response(self, phi):
        """Apply phase shift gate with memory-dependent phase θ
        
        Input state: |ψ_in⟩ = 1/√2(|0⟩ + e^(iφ/2)|1⟩)
        Phase shift: U(θ) = |0⟩⟨0| + e^(iθ)|1⟩⟨1|
        Output state: |ψ_out⟩ = 1/√2(|0⟩ + e^(i(φ/2 + θ))|1⟩)
        """
        # Memory affects the phase shift applied
        effective_theta = self.memory * self.params.memory_coupling
        
        # Calculate output ⟨σ_y⟩
        output_sigma_y = self.calculate_sigma_y(phi, effective_theta)
        
        return output_sigma_y
    
    def update_memory(self, input_signal, dt):
        """Update phase shift θ based on input signal
        
        The memory evolves based on the input ⟨σ_y⟩
        """
        # Memory update: θ evolves based on input signal magnitude
        dmemory = self.params.eta * input_signal - self.params.gamma * self.memory
        self.memory += dmemory * dt
        
        # Track integrated input
        self.integrated_input += input_signal * dt
    
    def simulate_hysteresis(self, num_cycles=2, points_per_cycle=2000):
        """Simulate the quantum memristor with phase sweep
        
        Sweep: φ: 0.025 → π → -π → 0.025
        """
        # Create phase sweep array
        phi_forward = np.linspace(self.params.phi_initial, np.pi, points_per_cycle//4)
        phi_reverse = np.linspace(np.pi, -np.pi, points_per_cycle//2)
        phi_return = np.linspace(-np.pi, self.params.phi_initial, points_per_cycle//4)
        
        # Complete cycle
        phi_single_cycle = np.concatenate([phi_forward, phi_reverse, phi_return])
        
        # Repeat for multiple cycles
        phi_array = np.tile(phi_single_cycle, num_cycles)
        
        # Time array
        T = num_cycles * 1.0  # Total time
        t_array = np.linspace(0, T, len(phi_array))
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        input_sigma_y = []
        output_sigma_y = []
        memory_values = []
        
        # Reset state
        self.reset()
        
        # Simulate
        for i, phi in enumerate(phi_array):
            # Calculate input ⟨σ_y⟩
            input_signal = self.calculate_sigma_y(phi, theta=0)
            
            # Get output after memory-dependent phase shift
            output_signal = self.memory_dependent_response(phi)
            
            # Update memory based on input
            self.update_memory(input_signal, dt)
            
            # Store values
            input_sigma_y.append(input_signal)
            output_sigma_y.append(output_signal)
            memory_values.append(self.memory)
        
        return {
            't': t_array,
            'phi': phi_array,
            'input': np.array(input_sigma_y),    # ⟨σ_y⟩ for input
            'output': np.array(output_sigma_y),  # ⟨σ_y⟩ for output
            'memory': np.array(memory_values)    # θ(t)
        }
    
    def simulate_no_memory(self, num_cycles=2, points_per_cycle=2000):
        """Simulate without memory for comparison"""
        # Save original parameters
        original_eta = self.params.eta
        original_gamma = self.params.gamma
        
        # Set no memory
        self.params.eta = 0.0
        self.params.gamma = 1000.0  # Very fast decay
        
        # Run simulation
        result = self.simulate_hysteresis(num_cycles, points_per_cycle)
        
        # Restore parameters
        self.params.eta = original_eta
        self.params.gamma = original_gamma
        
        return result

def plot_main_hysteresis(params=None, save=True):
    """Plot the main hysteresis loop - Input σ_y vs Output σ_y"""
    if params is None:
        params = PhaseShiftMemristorParams()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Initialize and simulate
    memristor = PhaseShiftQuantumMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=2000)
    data_no_mem = memristor.simulate_no_memory(num_cycles=3, points_per_cycle=2000)
    
    # Get last cycle
    n_points = len(data_mem['t'])
    last_cycle_start = int(2 * n_points / 3)
    
    # Extract last cycle
    input_mem = data_mem['input'][last_cycle_start:]
    output_mem = data_mem['output'][last_cycle_start:]
    input_no_mem = data_no_mem['input'][last_cycle_start:]
    output_no_mem = data_no_mem['output'][last_cycle_start:]
    
    # Smooth the output
    if len(output_mem) > 51:
        output_mem_smooth = savgol_filter(output_mem, 51, 3)
    else:
        output_mem_smooth = output_mem
    
    # Time for coloring
    t_last = data_mem['t'][last_cycle_start:] - data_mem['t'][last_cycle_start]
    colors = plt.cm.viridis(t_last / t_last[-1])
    
    # Plot with memory - color gradient
    for i in range(0, len(input_mem)-1, 10):
        ax.plot(input_mem[i:i+2], output_mem_smooth[i:i+2], 
                color=colors[i], linewidth=6, alpha=0.9)
    
    # Plot without memory
    ax.plot(input_no_mem, output_no_mem, 'r--', 
             linewidth=4, alpha=0.8, label='No memory (θ=0)')
    
    # Add direction arrows
    n_arrows = 6
    arrow_indices = np.linspace(100, len(input_mem)-100, n_arrows, dtype=int)
    for idx in arrow_indices:
        if idx + 20 < len(input_mem):
            ax.annotate('', xy=(input_mem[idx+20], output_mem_smooth[idx+20]),
                        xytext=(input_mem[idx], output_mem_smooth[idx]),
                        arrowprops=dict(arrowstyle='->', color='darkblue', lw=4, alpha=0.8))
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax.set_title('Phase Shift Quantum Memristor:\nσ_y Hysteresis', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.legend(loc='upper left')
    
    # Set axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_hysteresis.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_hysteresis.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_phase_evolution(params=None, save=True):
    """Plot the evolution of input phase φ and memory θ"""
    if params is None:
        params = PhaseShiftMemristorParams()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
    
    # Plot phase φ
    ax1.plot(data['t'], data['phi'], 'b-', linewidth=4)
    ax1.set_ylabel('Input Phase φ (rad)', fontweight='bold')
    ax1.set_title('Phase Evolution in Quantum Memristor', fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_ylim(-np.pi-0.5, np.pi+0.5)
    
    # Add π markers
    ax1.axhline(y=np.pi, color='r', linestyle=':', alpha=0.5, linewidth=2)
    ax1.axhline(y=-np.pi, color='r', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(0.02, np.pi+0.1, 'π', transform=ax1.get_yaxis_transform(), fontsize=20)
    ax1.text(0.02, -np.pi-0.1, '-π', transform=ax1.get_yaxis_transform(), fontsize=20)
    
    # Plot memory θ
    ax2.plot(data['t'], data['memory'], 'g-', linewidth=5)
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Memory Phase θ (rad)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_evolution.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_evolution.png', bbox_inches='tight', dpi=300)
    
    return fig, (ax1, ax2)

def plot_sigma_y_evolution(params=None, save=True):
    """Plot time evolution of input and output ⟨σ_y⟩"""
    if params is None:
        params = PhaseShiftMemristorParams()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
    
    ax.plot(data['t'], data['output'], 'b-', linewidth=4, label='Output ⟨σ_y⟩')
    ax.plot(data['t'], data['input'], 'k--', linewidth=3.5, alpha=0.7, label='Input ⟨σ_y⟩')
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('⟨σ_y⟩', fontweight='bold')
    ax.set_title('Pauli-Y Expectation Value Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_sigma_y.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_sigma_y.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_3d_trajectory(params=None, save=True):
    """Plot 3D trajectory showing input ⟨σ_y⟩, output ⟨σ_y⟩, and θ"""
    if params is None:
        params = PhaseShiftMemristorParams()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=2000)
    
    n_points = len(data['t'])
    last_cycle_start = int(2 * n_points / 3)
    
    input_mem = data['input'][last_cycle_start:]
    output_mem = data['output'][last_cycle_start:]
    
    if len(output_mem) > 51:
        output_mem_smooth = savgol_filter(output_mem, 51, 3)
    else:
        output_mem_smooth = output_mem
    
    ax.plot(input_mem[::20], output_mem_smooth[::20], 
             data['memory'][last_cycle_start::20], 
             'b-', linewidth=4, alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontweight='bold', labelpad=15)
    ax.set_ylabel('Output ⟨σ_y⟩', fontweight='bold', labelpad=15)
    ax.set_zlabel('Memory θ (rad)', fontweight='bold', labelpad=15)
    ax.set_title('3D Phase Space of Phase Shift Memristor', fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_3d.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_3d.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_parameter_effects(save=True):
    """Plot effects of different parameters on hysteresis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Effect of eta (feedback strength)
    eta_values = [0.2, 0.3, 0.4, 0.5, 0.6]
    colors1 = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    for i, eta in enumerate(eta_values):
        params = PhaseShiftMemristorParams(eta=eta)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        ax1.plot(data['input'][start:], data['output'][start:], 
                label=f'η={eta}', linewidth=4, color=colors1[i])
    
    ax1.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax1.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax1.set_title('Effect of Feedback Strength η', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    
    # Effect of gamma (relaxation rate)
    gamma_values = [0.005, 0.01, 0.02, 0.04, 0.08]
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(gamma_values)))
    
    for i, gamma in enumerate(gamma_values):
        params = PhaseShiftMemristorParams(gamma=gamma)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        ax2.plot(data['input'][start:], data['output'][start:], 
                label=f'γ={gamma}', linewidth=4, color=colors2[i])
    
    ax2.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax2.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax2.set_title('Effect of Relaxation Rate γ', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    
    # Effect of memory coupling
    coupling_values = [0.4, 0.6, 0.8, 1.0, 1.2]
    colors3 = plt.cm.cool(np.linspace(0, 1, len(coupling_values)))
    
    for i, coupling in enumerate(coupling_values):
        params = PhaseShiftMemristorParams(memory_coupling=coupling)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        ax3.plot(data['input'][start:], data['output'][start:], 
                label=f'κ={coupling}', linewidth=4, color=colors3[i])
    
    ax3.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax3.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax3.set_title('Effect of Memory Coupling κ', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    
    # Phase space plot for default parameters
    params = PhaseShiftMemristorParams()
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=2000)
    
    n_points = len(data['t'])
    last_cycle_start = int(2 * n_points / 3)
    
    t_last = data['t'][last_cycle_start:] - data['t'][last_cycle_start]
    scatter = ax4.scatter(data['memory'][last_cycle_start:], 
                         data['input'][last_cycle_start:], 
                         c=t_last, cmap='plasma', s=4, alpha=0.6)
    ax4.set_xlabel('Memory θ (rad)', fontweight='bold')
    ax4.set_ylabel('Input ⟨σ_y⟩', fontweight='bold')
    ax4.set_title('θ-⟨σ_y⟩ Phase Space', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_parameters.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_parameters.png', bbox_inches='tight', dpi=300)
    
    return fig, ((ax1, ax2), (ax3, ax4))

# Run all plots
if __name__ == "__main__":
    print("Generating Phase Shift Quantum Memristor Figures...")
    print("=" * 60)
    print("Operating in XY plane with σ_y observables")
    print("Initial state: |ψ⟩ = 1/√2(|0⟩ + e^(iφ/2)|1⟩)")
    print("Phase sweep: φ: 0.025 → π → -π → 0.025")
    print("=" * 60)
    
    # Create all figures
    print("\n1. Main hysteresis loop (⟨σ_y⟩ vs ⟨σ_y⟩)...")
    plot_main_hysteresis()
    
    print("2. Phase evolution (φ and θ vs time)...")
    plot_phase_evolution()
    
    print("3. σ_y expectation value evolution...")
    plot_sigma_y_evolution()
    
    print("4. 3D trajectory...")
    plot_3d_trajectory()
    
    print("5. Parameter effects...")
    plot_parameter_effects()
    
    print("\n" + "=" * 60)
    print("All figures saved successfully!")
    print("\nGenerated files:")
    print("  - phase_memristor_hysteresis.pdf/.png")
    print("  - phase_memristor_evolution.pdf/.png")
    print("  - phase_memristor_sigma_y.pdf/.png")
    print("  - phase_memristor_3d.pdf/.png")
    print("  - phase_memristor_parameters.pdf/.png")
    
    # Calculate and display key characteristics
    params = PhaseShiftMemristorParams()
    print(f"\nKey parameters:")
    print(f"  Initial phase φ₀ = {params.phi_initial:.3f} rad")
    print(f"  Feedback strength η = {params.eta}")
    print(f"  Relaxation rate γ = {params.gamma}")
    print(f"  Memory coupling κ = {params.memory_coupling}")
    
    print("\nPhysical interpretation:")
    print("  - Memory parameter θ represents accumulated phase shift")
    print("  - Input/output observables are Pauli-Y expectation values")
    print("  - Hysteresis emerges from memory-dependent phase accumulation")
    
    # Show all plots
    plt.show()
