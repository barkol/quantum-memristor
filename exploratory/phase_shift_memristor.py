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
class PhaseMemristorParams:
    """Parameters for quantum memristor with phase shift gate
    
    Memory parameter θ is the phase shift applied by the unitary gate
    """
    # Memory dynamics (phase shift evolution)
    eta: float = 1.0           # Feedback strength
    gamma: float = 0.05        # Relaxation rate for memory retention
    
    # Memory coupling parameters
    memory_coupling: float = 1.0  # How strongly memory affects output
    
    # Initial phase
    initial_phi: float = 0.0  # Initial phase in |+⟩_φ state

class QuantumPhaseMemristor:
    """Quantum memristor using phase shift gate with memory-dependent phase θ"""
    
    def __init__(self, params: PhaseMemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (phase shift θ = 0)"""
        self.memory = 0.0  # θ = 0 (phase shift in radians)
        self.integrated_input = 0.0
    
    def phase_shift_transformation(self, phi: float, theta: float) -> float:
        """Apply phase shift gate U_θ to state |+⟩_φ
        
        Input state: |+⟩_φ = 1/√2(|0⟩ + e^(iφ)|1⟩)
        After U_θ: |+⟩_{φ+θ} = 1/√2(|0⟩ + e^(i(φ+θ))|1⟩)
        
        Returns the phase of the output state
        """
        return phi + theta * self.params.memory_coupling
    
    def calculate_sigma_y(self, phase: float) -> float:
        """Calculate ⟨σ_y⟩ for state |+⟩_φ = 1/√2(|0⟩ + e^(iφ)|1⟩)
        
        ⟨σ_y⟩ = sin(φ)
        """
        return np.sin(phase)
    
    def update_memory(self, input_observable: float, dt: float):
        """Update phase shift θ based on measurement feedback
        
        Memory evolves based on the input observable value
        """
        # Memory update with feedback proportional to input
        dmemory = self.params.eta * input_observable - self.params.gamma * self.memory
        self.memory += dmemory * dt
        
        # Track integrated input
        self.integrated_input += input_observable * dt
    
    def simulate_hysteresis(self, num_cycles=3, points_per_cycle=2000):
        """Simulate the quantum memristor with phase sweep"""
        
        # Create phase sweep: 0.025 → π → -π → 0.025
        points_up = points_per_cycle // 3
        points_down = points_per_cycle // 3
        points_return = points_per_cycle - points_up - points_down
        
        # Single cycle phase sweep
        phi_up = np.linspace(self.params.initial_phi, np.pi, points_up)
        phi_down = np.linspace(np.pi, -np.pi, points_down)
        phi_return = np.linspace(-np.pi, self.params.initial_phi, points_return)
        phi_cycle = np.concatenate([phi_up, phi_down, phi_return])
        
        # Repeat for multiple cycles
        phi_array = np.tile(phi_cycle, num_cycles)
        
        # Time array
        t_array = np.linspace(0, num_cycles * 10, len(phi_array))  # 10 seconds per cycle
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        input_sigma_y = []
        output_sigma_y = []
        memory_values = []
        output_phases = []
        
        # Reset state
        self.reset()
        
        # Simulate
        for i, phi in enumerate(phi_array):
            # Calculate input observable
            input_obs = self.calculate_sigma_y(phi)
            
            # Apply phase shift transformation
            output_phase = self.phase_shift_transformation(phi, self.memory)
            
            # Calculate output observable
            output_obs = self.calculate_sigma_y(output_phase)
            
            # Update memory based on input
            self.update_memory(input_obs, dt)
            
            # Store values
            input_sigma_y.append(input_obs)
            output_sigma_y.append(output_obs)
            memory_values.append(self.memory)
            output_phases.append(output_phase)
        
        return {
            't': t_array,
            'phi': phi_array,
            'input_sigma_y': np.array(input_sigma_y),
            'output_sigma_y': np.array(output_sigma_y),
            'memory': np.array(memory_values),
            'output_phase': np.array(output_phases)
        }
    
    def simulate_no_memory(self, num_cycles=3, points_per_cycle=2000):
        """Simulate without memory for comparison"""
        # Save original parameters
        original_eta = self.params.eta
        original_gamma = self.params.gamma
        original_coupling = self.params.memory_coupling
        
        # Set no memory
        self.params.eta = 0.0
        self.params.gamma = 1000.0  # Very fast decay
        self.params.memory_coupling = 0.0  # No coupling
        
        # Run simulation
        result = self.simulate_hysteresis(num_cycles, points_per_cycle)
        
        # Restore parameters
        self.params.eta = original_eta
        self.params.gamma = original_gamma
        self.params.memory_coupling = original_coupling
        
        return result

def plot_main_hysteresis(params=None, save=True):
    """Plot the main hysteresis loop - Input σ_y vs Output σ_y"""
    if params is None:
        params = PhaseMemristorParams()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Initialize and simulate
    memristor = QuantumPhaseMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=3000)
    data_no_mem = memristor.simulate_no_memory(num_cycles=4, points_per_cycle=3000)
    
    # Get last cycle
    n_points = len(data_mem['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    # Extract last cycle
    input_mem = data_mem['input_sigma_y'][last_cycle_start:]
    output_mem = data_mem['output_sigma_y'][last_cycle_start:]
    input_no_mem = data_no_mem['input_sigma_y'][last_cycle_start:]
    output_no_mem = data_no_mem['output_sigma_y'][last_cycle_start:]
    
    # Time for coloring
    t_last = data_mem['t'][last_cycle_start:] - data_mem['t'][last_cycle_start]
    colors = plt.cm.viridis(t_last / t_last[-1])
    
    # Plot with memory - color gradient
    for i in range(0, len(input_mem)-1, 20):
        ax.plot(input_mem[i:i+2], output_mem[i:i+2], 
                color=colors[i], linewidth=6, alpha=0.9)
    
    # Plot without memory
    ax.plot(input_no_mem, output_no_mem, 'r--', 
            linewidth=4, alpha=0.8, label='No memory (θ=0)')
    
    # Add direction arrows
    n_arrows = 8
    arrow_indices = np.linspace(100, len(input_mem)-100, n_arrows, dtype=int)
    for idx in arrow_indices:
        if idx + 30 < len(input_mem):
            arrow_color = 'darkgreen' if output_mem[idx] > input_mem[idx] else 'darkblue'
            ax.annotate('', xy=(input_mem[idx+30], output_mem[idx+30]),
                        xytext=(input_mem[idx], output_mem[idx]),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=4, alpha=0.8))
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax.set_title('Quantum Phase-Shift Memristor:\nσ_y Hysteresis', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2.5)
    ax.legend(loc='upper left')
    
    # Set axis limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Add text annotations
    ax.text(0.98, 0.02, f'η={params.eta}, γ={params.gamma}', 
            transform=ax.transAxes, fontsize=18,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            ha='right', va='bottom')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_hysteresis.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_hysteresis.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_phase_evolution(params=None, save=True):
    """Plot phase φ evolution over time"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    memristor = QuantumPhaseMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=2000)
    
    # Plot phase evolution
    ax1.plot(data['t'], data['phi'], 'k-', linewidth=3, label='Input phase φ')
    ax1.plot(data['t'], data['output_phase'], 'b-', linewidth=3, label='Output phase (φ+θ)')
    ax1.set_ylabel('Phase (rad)', fontweight='bold')
    ax1.set_title('Phase Evolution in Quantum Memristor', fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(y=np.pi, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='π')
    ax1.axhline(y=-np.pi, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='-π')
    
    # Plot observables
    ax2.plot(data['t'], data['input_sigma_y'], 'r-', linewidth=3, label='Input ⟨σ_y⟩')
    ax2.plot(data['t'], data['output_sigma_y'], 'g-', linewidth=3, label='Output ⟨σ_y⟩')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('⟨σ_y⟩', fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.legend()
    ax2.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_evolution.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_evolution.png', bbox_inches='tight', dpi=300)
    
    return fig, (ax1, ax2)

def plot_memory_evolution(params=None, save=True):
    """Plot memory parameter θ evolution"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    memristor = QuantumPhaseMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=2000)
    
    ax.plot(data['t'], data['memory'], 'purple', linewidth=5)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Memory Parameter θ (rad)', fontweight='bold')
    ax.set_title('Phase Shift Memory Evolution', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add annotations
    ax.text(0.02, 0.98, f'Maximum |θ| ≈ {np.max(np.abs(data["memory"])):.2f} rad', 
            transform=ax.transAxes, fontsize=20,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            va='top')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_memory.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_memory.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_phase_space_3d(params=None, save=True):
    """Plot 3D trajectory showing input σ_y, output σ_y, and θ"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    memristor = QuantumPhaseMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=2000)
    
    n_points = len(data['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    # Downsample for clarity
    step = 30
    ax.plot(data['input_sigma_y'][last_cycle_start::step], 
            data['output_sigma_y'][last_cycle_start::step], 
            data['memory'][last_cycle_start::step], 
            'b-', linewidth=4, alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontweight='bold', labelpad=15)
    ax.set_ylabel('Output ⟨σ_y⟩', fontweight='bold', labelpad=15)
    ax.set_zlabel('Memory θ (rad)', fontweight='bold', labelpad=15)
    ax.set_title('3D Phase Space of Quantum Memristor', fontweight='bold', pad=20)
    ax.view_init(elev=20, azim=45)
    
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_3d.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_3d.png', bbox_inches='tight', dpi=300)
    
    return fig, ax

def plot_parameter_effects(save=True):
    """Plot effects of different parameters"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Effect of feedback strength η
    eta_values = [0.3, 0.5, 0.8, 1.0, 1.2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    for i, eta in enumerate(eta_values):
        params = PhaseMemristorParams(eta=eta)
        memristor = QuantumPhaseMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1500)
        
        n = len(data['t'])
        start = n // 2
        ax1.plot(data['input_sigma_y'][start:], data['output_sigma_y'][start:], 
                label=f'η={eta}', linewidth=4, color=colors[i])
    
    ax1.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax1.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax1.set_title('Effect of Feedback Strength η', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    
    # Effect of relaxation rate γ
    gamma_values = [0.005, 0.01, 0.02, 0.05, 0.1]
    colors = plt.cm.plasma(np.linspace(0, 1, len(gamma_values)))
    
    for i, gamma in enumerate(gamma_values):
        params = PhaseMemristorParams(gamma=gamma)
        memristor = QuantumPhaseMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1500)
        
        n = len(data['t'])
        start = n // 2
        ax2.plot(data['input_sigma_y'][start:], data['output_sigma_y'][start:], 
                label=f'γ={gamma}', linewidth=4, color=colors[i])
    
    ax2.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax2.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax2.set_title('Effect of Relaxation Rate γ', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    
    # Effect of memory coupling
    coupling_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.cool(np.linspace(0, 1, len(coupling_values)))
    
    for i, coupling in enumerate(coupling_values):
        params = PhaseMemristorParams(memory_coupling=coupling)
        memristor = QuantumPhaseMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1500)
        
        n = len(data['t'])
        start = n // 2
        ax3.plot(data['input_sigma_y'][start:], data['output_sigma_y'][start:], 
                label=f'κ={coupling}', linewidth=4, color=colors[i])
    
    ax3.set_xlabel('Input ⟨σ_y⟩', fontweight='bold')
    ax3.set_ylabel('Output ⟨σ_y⟩', fontweight='bold')
    ax3.set_title('Effect of Memory Coupling κ', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    
    # Loop area analysis
    ax4.axis('off')
    summary = """
    Quantum Phase-Shift Memristor
    
    Key Features:
    • Input state: |+⟩_φ with phase sweep
    • Memory: Phase shift θ(t)
    • Observables: ⟨σ_y⟩ = sin(φ)
    
    Optimal Parameters:
    • η = 0.8 (feedback strength)
    • γ = 0.02 (relaxation rate)
    • κ = 0.6 (memory coupling)
    
    Physics:
    • U_θ|+⟩_φ = |+⟩_{φ+θ}
    • Clear hysteresis loops
    • Memory accumulation
    """
    ax4.text(0.1, 0.5, summary, fontsize=20, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_parameters.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_parameters.png', bbox_inches='tight', dpi=300)
    
    return fig

# Run all plots
if __name__ == "__main__":
    print("Generating Quantum Phase-Shift Memristor Figures...")
    print("=" * 60)
    print("Configuration:")
    print("- Initial state: |+⟩_φ with φ starting at 0.025")
    print("- Phase sweep: 0.025 → π → -π → 0.025")
    print("- Observables: Input and Output ⟨σ_y⟩")
    print("- Memory: Phase shift θ(t)")
    print("=" * 60)
    
    # Create all figures
    print("\n1. Main hysteresis loop...")
    plot_main_hysteresis()
    
    print("2. Phase evolution...")
    plot_phase_evolution()
    
    print("3. Memory evolution...")
    plot_memory_evolution()
    
    print("4. 3D phase space...")
    plot_phase_space_3d()
    
    print("5. Parameter effects...")
    plot_parameter_effects()
    
    print("\n" + "=" * 60)
    print("All figures saved successfully!")
    print("\nGenerated files:")
    print("  - phase_memristor_hysteresis.pdf/.png")
    print("  - phase_memristor_evolution.pdf/.png")
    print("  - phase_memristor_memory.pdf/.png")
    print("  - phase_memristor_3d.pdf/.png")
    print("  - phase_memristor_parameters.pdf/.png")
    
    # Show plots
    plt.show()