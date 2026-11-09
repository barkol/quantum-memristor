import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D

# Configure plotting
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

@dataclass
class PhaseMemristorParams:
    """Parameters for phase-shift quantum memristor
    
    The memory parameter θ is the phase shift angle that evolves based on detection events
    """
    # Memory dynamics (phase θ evolution)
    eta: float = 0.8           # Feedback strength (how much detection affects θ)
    gamma: float = 0.05        # Relaxation rate (memory decay)
    theta_0: float = 0.0       # Equilibrium phase (where θ relaxes to)
    
    # Input modulation parameters
    sweep_rate: float = 0.15   # Frequency of input phase modulation
    amplitude: float = np.pi   # Amplitude of input phase modulation (full range)
    
    # Detection parameters
    detection_gain: float = 0.9  # Scaling factor for detection probability

class PhaseShiftQuantumMemristor:
    """Quantum memristor using phase shift gate with memory-dependent phase θ"""
    
    def __init__(self, params: PhaseMemristorParams):
        self.params = params
        self.reset()
    
    def reset(self):
        """Reset to initial state (θ = 0)"""
        self.theta = 0.0  # Memory phase shift
        self.integrated_detections = 0.0
    
    def phase_shift_transformation(self, input_phase: float, theta: float) -> dict:
        """Apply phase shift gate with memory phase θ to input state
        
        Input state: |ψ_in⟩ = 1/√2(|0⟩ + e^(i*input_phase)|1⟩)
        After phase shift: |ψ_out⟩ = 1/√2(|0⟩ + e^(i*(input_phase + theta))|1⟩)
        
        Returns:
            dict with 'sigma_x' and 'sigma_y' expectation values
        """
        # Total phase after memory-dependent phase shift
        total_phase = input_phase + theta
        
        # Calculate expectation values
        # For |ψ⟩ = 1/√2(|0⟩ + e^(iφ)|1⟩):
        # ⟨σ_x⟩ = cos(φ)
        # ⟨σ_y⟩ = -sin(φ)
        
        sigma_x_out = np.cos(total_phase)
        sigma_y_out = -np.sin(total_phase)
        
        return {
            'sigma_x': sigma_x_out,
            'sigma_y': sigma_y_out
        }
    
    def detection_probability(self, output_state: dict) -> float:
        """Calculate detection probability based on output state
        
        We use |⟨σ_x⟩| as a proxy for detection probability
        """
        return self.params.detection_gain * (1 + output_state['sigma_x']) / 2
    
    def update_memory(self, input_phase: float, detection_prob: float, dt: float):
        """Update memory phase θ based on detection probability and input
        
        The memory evolution depends on both the detection probability
        and the current input phase
        """
        # Memory update with input-dependent feedback
        # Detection rate modulated by input phase magnitude
        feedback_strength = self.params.eta * (1 + 0.3 * np.abs(np.sin(input_phase)))
        
        # Update θ based on detection probability
        dtheta = feedback_strength * detection_prob - self.params.gamma * (self.theta - self.params.theta_0)
        self.theta += dtheta * dt
        
        # Track integrated detections
        self.integrated_detections += detection_prob * dt
    
    def simulate_hysteresis(self, num_cycles=3, points_per_cycle=2000):
        """Simulate the quantum memristor to show σ_y vs σ_x hysteresis"""
        # Time array
        period = 2 * np.pi / self.params.sweep_rate
        T = num_cycles * period
        t_array = np.linspace(0, T, num_cycles * points_per_cycle)
        dt = t_array[1] - t_array[0]
        
        # Storage arrays
        input_phases = []
        input_sigma_y = []
        output_sigma_x = []
        output_sigma_y = []
        theta_values = []
        
        # Reset state
        self.reset()
        
        # Simulate
        for i, t in enumerate(t_array):
            # Sinusoidal input phase modulation
            input_phase = self.params.amplitude * np.sin(self.params.sweep_rate * t)
            
            # Calculate input observable
            sigma_y_in = -np.sin(input_phase)
            
            # Apply phase shift transformation
            output_state = self.phase_shift_transformation(input_phase, self.theta)
            
            # Get detection probability
            detection_prob = self.detection_probability(output_state)
            
            # Update memory based on detection
            self.update_memory(input_phase, detection_prob, dt)
            
            # Store values
            input_phases.append(input_phase)
            input_sigma_y.append(sigma_y_in)
            output_sigma_x.append(output_state['sigma_x'])
            output_sigma_y.append(output_state['sigma_y'])
            theta_values.append(self.theta)
        
        return {
            't': t_array,
            'input_phase': np.array(input_phases),
            'input_sigma_y': np.array(input_sigma_y),
            'output_sigma_x': np.array(output_sigma_x),
            'output_sigma_y': np.array(output_sigma_y),
            'theta': np.array(theta_values)
        }
    
    def simulate_no_memory(self, num_cycles=3, points_per_cycle=2000):
        """Simulate without memory for comparison (fixed θ = 0)"""
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
    """Plot the main hysteresis loop - σ_y (input) vs σ_x (output)"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize and simulate
    memristor = PhaseShiftQuantumMemristor(params)
    data_mem = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=2000)
    data_no_mem = memristor.simulate_no_memory(num_cycles=4, points_per_cycle=2000)
    
    # Get last cycle for clean loop
    n_points = len(data_mem['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    # Extract last cycle
    input_y_mem = data_mem['input_sigma_y'][last_cycle_start:]
    output_x_mem = data_mem['output_sigma_x'][last_cycle_start:]
    input_y_no_mem = data_no_mem['input_sigma_y'][last_cycle_start:]
    output_x_no_mem = data_no_mem['output_sigma_x'][last_cycle_start:]
    
    # Smooth the output if needed
    output_x_mem_smooth = savgol_filter(output_x_mem, 51, 3)
    
    # Time for coloring
    t_last = data_mem['t'][last_cycle_start:] - data_mem['t'][last_cycle_start]
    colors = plt.cm.viridis(t_last / t_last[-1])
    
    # Plot with memory - color gradient
    for i in range(0, len(input_y_mem)-1, 10):
        ax.plot(input_y_mem[i:i+2], output_x_mem_smooth[i:i+2], 
                color=colors[i], linewidth=3, alpha=0.9)
    
    # Plot without memory
    ax.plot(input_y_no_mem, output_x_no_mem, 'r--', 
            linewidth=2, alpha=0.8, label='No memory (θ=0)')
    
    # Add direction arrows
    n_arrows = 6
    arrow_indices = np.linspace(100, len(input_y_mem)-100, n_arrows, dtype=int)
    for idx in arrow_indices:
        if idx + 20 < len(input_y_mem):
            ax.annotate('', xy=(input_y_mem[idx+20], output_x_mem_smooth[idx+20]),
                        xytext=(input_y_mem[idx], output_x_mem_smooth[idx]),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2, alpha=0.6))
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontsize=14)
    ax.set_ylabel('Output ⟨σ_x⟩', fontsize=14)
    ax.set_title('Phase Shift Quantum Memristor: Hysteresis Loop', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Add text annotation
    ax.text(0.02, 0.98, 'Phase shift gate\nwith memory θ(t)', 
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_hysteresis.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_hysteresis.png', bbox_inches='tight', dpi=150)
    
    return fig, ax

def plot_time_evolution(params=None, save=True):
    """Plot time evolution of input and output observables"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=1000)
    
    # Plot observables
    ax1.plot(data['t'], data['input_sigma_y'], 'b-', linewidth=2, label='Input ⟨σ_y⟩')
    ax1.plot(data['t'], data['output_sigma_x'], 'r-', linewidth=2, label='Output ⟨σ_x⟩')
    ax1.set_ylabel('Expectation Value', fontsize=12)
    ax1.set_title('Quantum Observable Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-1.2, 1.2)
    
    # Plot phase evolution
    ax2.plot(data['t'], data['input_phase'], 'g--', linewidth=2, label='Input phase φ')
    ax2.plot(data['t'], data['theta'], 'purple', linewidth=2, label='Memory phase θ')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Phase (rad)', fontsize=12)
    ax2.set_title('Phase Evolution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_time_evolution.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_time_evolution.png', bbox_inches='tight', dpi=150)
    
    return fig, (ax1, ax2)

def plot_memory_evolution(params=None, save=True):
    """Plot memory phase θ evolution"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=3, points_per_cycle=1000)
    
    ax.plot(data['t'], data['theta'], 'purple', linewidth=3)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Memory Phase θ (rad)', fontsize=12)
    ax.set_title('Phase Shift Memory Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add annotation
    ax.text(0.02, 0.95, f'η = {params.eta} (feedback)\nγ = {params.gamma} (relaxation)', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_memory.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_memory.png', bbox_inches='tight', dpi=150)
    
    return fig, ax

def plot_phase_space(params=None, save=True):
    """Plot phase space: input phase vs memory phase"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=2000)
    
    n_points = len(data['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    input_phase = data['input_phase'][last_cycle_start:]
    theta = data['theta'][last_cycle_start:]
    t_last = data['t'][last_cycle_start:] - data['t'][last_cycle_start]
    
    scatter = ax.scatter(input_phase, theta, c=t_last, cmap='plasma', s=2, alpha=0.6)
    ax.set_xlabel('Input Phase φ (rad)', fontsize=12)
    ax.set_ylabel('Memory Phase θ (rad)', fontsize=12)
    ax.set_title('Phase Space: Input vs Memory', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_phase_space.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_phase_space.png', bbox_inches='tight', dpi=150)
    
    return fig, ax

def plot_3d_trajectory(params=None, save=True):
    """Plot 3D trajectory: σ_y vs σ_x vs θ"""
    if params is None:
        params = PhaseMemristorParams()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    memristor = PhaseShiftQuantumMemristor(params)
    data = memristor.simulate_hysteresis(num_cycles=4, points_per_cycle=2000)
    
    n_points = len(data['t'])
    last_cycle_start = int(3 * n_points / 4)
    
    # Downsample for clarity
    step = 20
    sigma_y = data['input_sigma_y'][last_cycle_start::step]
    sigma_x = data['output_sigma_x'][last_cycle_start::step]
    theta = data['theta'][last_cycle_start::step]
    
    ax.plot(sigma_y, sigma_x, theta, 'b-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Input ⟨σ_y⟩', fontsize=10)
    ax.set_ylabel('Output ⟨σ_x⟩', fontsize=10)
    ax.set_zlabel('Memory θ (rad)', fontsize=10)
    ax.set_title('3D Phase Space Trajectory', fontsize=14)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_3d.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_3d.png', bbox_inches='tight', dpi=150)
    
    return fig, ax

def plot_parameter_effects(save=True):
    """Plot effects of varying key parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parameter Effects on Phase Shift Memristor', fontsize=16)
    
    # Effect of feedback strength η
    ax = axes[0, 0]
    eta_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    for i, eta in enumerate(eta_values):
        params = PhaseMemristorParams(eta=eta)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        ax.plot(data['input_sigma_y'][start:], data['output_sigma_x'][start:], 
                label=f'η={eta}', linewidth=2, color=colors[i])
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_x⟩')
    ax.set_title('Effect of Feedback Strength η')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Effect of relaxation rate γ
    ax = axes[0, 1]
    gamma_values = [0.01, 0.03, 0.05, 0.1, 0.2]
    colors = plt.cm.plasma(np.linspace(0, 1, len(gamma_values)))
    
    for i, gamma in enumerate(gamma_values):
        params = PhaseMemristorParams(gamma=gamma)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        ax.plot(data['input_sigma_y'][start:], data['output_sigma_x'][start:], 
                label=f'γ={gamma}', linewidth=2, color=colors[i])
    
    ax.set_xlabel('Input ⟨σ_y⟩')
    ax.set_ylabel('Output ⟨σ_x⟩')
    ax.set_title('Effect of Relaxation Rate γ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Loop area vs η
    ax = axes[1, 0]
    eta_range = np.linspace(0.1, 1.5, 20)
    loop_areas = []
    
    for eta in eta_range:
        params = PhaseMemristorParams(eta=eta)
        memristor = PhaseShiftQuantumMemristor(params)
        data = memristor.simulate_hysteresis(num_cycles=2, points_per_cycle=1000)
        
        n = len(data['t'])
        start = n // 2
        x = data['input_sigma_y'][start:]
        y = data['output_sigma_x'][start:]
        
        # Calculate loop area using shoelace formula
        area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
        loop_areas.append(area)
    
    ax.plot(eta_range, loop_areas, 'b-', linewidth=2)
    ax.set_xlabel('Feedback Strength η')
    ax.set_ylabel('Loop Area')
    ax.set_title('Hysteresis Loop Area vs η')
    ax.grid(True, alpha=0.3)
    
    # Memory phase range vs parameters
    ax = axes[1, 1]
    ax.text(0.1, 0.5, 
            'Phase Shift Memristor Properties:\n\n'
            '• Memory parameter: phase shift θ\n'
            '• Input state: |+⟩ with phase φ(t)\n'
            '• Transformation: U(θ)|ψ⟩\n'
            '• Input observable: ⟨σ_y⟩ = -sin(φ)\n'
            '• Output observable: ⟨σ_x⟩ = cos(φ+θ)\n\n'
            'Optimal parameters:\n'
            '• η = 0.6-1.0 (strong memory)\n'
            '• γ = 0.03-0.08 (moderate decay)\n'
            '• Wide hysteresis loops achieved',
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    
    plt.tight_layout()
    
    if save:
        plt.savefig('phase_memristor_parameters.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('phase_memristor_parameters.png', bbox_inches='tight', dpi=150)
    
    return fig

# Run all simulations
if __name__ == "__main__":
    print("Generating Phase Shift Quantum Memristor Simulations...")
    print("=" * 60)
    
    # Create all figures
    print("\n1. Main hysteresis loop (σ_y vs σ_x)...")
    fig1, ax1 = plot_main_hysteresis()
    
    print("2. Time evolution of observables...")
    fig2, axes2 = plot_time_evolution()
    
    print("3. Memory phase evolution...")
    fig3, ax3 = plot_memory_evolution()
    
    print("4. Phase space visualization...")
    fig4, ax4 = plot_phase_space()
    
    print("5. 3D trajectory...")
    fig5, ax5 = plot_3d_trajectory()
    
    print("6. Parameter effects...")
    fig6 = plot_parameter_effects()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print("\nKey features of the phase shift quantum memristor:")
    print("- Memory parameter θ is the phase shift angle")
    print("- Initial state: |+⟩ = 1/√2(|0⟩ + |1⟩)")
    print("- Input modulation creates varying ⟨σ_y⟩")
    print("- Output ⟨σ_x⟩ depends on both input and memory phases")
    print("- Clear hysteresis loops demonstrate memory effects")
    
    # Show all plots
    plt.show()
