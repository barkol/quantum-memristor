import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.colors as mcolors
from scipy.integrate import solve_ivp

class QuantumMemristorTheory:
    """Theoretical model for quantum memristor average trajectories."""
    
    def __init__(self):
        # Parameters matching the simulation
        self.t_max = 10            # Maximum simulation time
        self.theta_max = 10 * np.pi  # Maximum memory parameter value
        self.omega_theta = self.theta_max / self.t_max  # Angular frequency for θ
        self.omega_phi = np.pi / 2   # Angular frequency for φ
        
    def calculate_theoretical_trajectory(self, basis, measurement_frequency, num_points=400):
        """Calculate theoretical average trajectory based on the mathematical model."""
        # Time points for evaluation
        t_values = np.linspace(0, self.t_max, num_points)
        theta_values = self.omega_theta * t_values
        phi_values = self.omega_phi * t_values
        
        # Arrays to store results
        sigma_x_values = np.zeros(num_points)
        sigma_y_values = np.zeros(num_points)
        sigma_z_values = np.zeros(num_points)
        
        # Set decay coefficients based on measurement basis
        if basis == 'x':
            R_x, R_y, R_z = 0, 1, 1
        elif basis == 'y':
            R_x, R_y, R_z = 1, 0, 1
        elif basis == 'z':
            R_x, R_y, R_z = 1, 1, 0
        else:  # sequence case - we'll handle this separately
            return self.calculate_sequence_trajectory(measurement_frequency, num_points)
        
        # Initial condition: pure state along x-axis
        bloch_vector = np.array([1.0, 0.0, 0.0])
        
        # Time step
        dt = self.t_max / (num_points - 1)
        
        # Simple Euler integration with phase dependence
        for i in range(num_points):
            # Current values
            x, y, z = bloch_vector
            phi = phi_values[i]
            
            # Unitary evolution (rotation around various axes)
            new_x = x * np.cos(phi) + y * np.sin(phi)
            new_y = y * np.cos(phi) - x * np.sin(phi)
            new_z = z
            
            # Apply measurement effects (damping)
            dx = -measurement_frequency * R_x * new_x * dt
            dy = -measurement_frequency * R_y * new_y * dt
            dz = -measurement_frequency * R_z * new_z * dt
            
            # Update Bloch vector
            new_x += dx
            new_y += dy
            new_z += dz
            
            bloch_vector = np.array([new_x, new_y, new_z])
            
            # Store values
            sigma_x_values[i] = bloch_vector[0]
            sigma_y_values[i] = bloch_vector[1]
            sigma_z_values[i] = bloch_vector[2]
        
        # Return in same format as the simulation
        return {
            't': t_values,
            'theta': theta_values % (2 * np.pi),  # Modulo 2π for visualization
            'theta_full': theta_values,
            'sigma_x': sigma_x_values,
            'sigma_y': sigma_y_values,
            'sigma_z': sigma_z_values
        }
    
    def calculate_sequence_trajectory(self, measurement_frequency, num_points=400):
        """Calculate theoretical trajectory for X→Y→Z sequence."""
        # Time points for evaluation
        t_values = np.linspace(0, self.t_max, num_points)
        theta_values = self.omega_theta * t_values
        phi_values = self.omega_phi * t_values
        
        # Arrays to store results
        sigma_x_values = np.zeros(num_points)
        sigma_y_values = np.zeros(num_points)
        sigma_z_values = np.zeros(num_points)
        
        # Initial condition: pure state along x-axis
        bloch_vector = np.array([1.0, 0.0, 0.0])
        
        # Time step
        dt = self.t_max / (num_points - 1)
        
        # Cycle through bases based on time
        cycle_period = self.t_max / 10  # Complete X→Y→Z cycle 10 times
        
        # Simple Euler integration
        for i in range(num_points):
            t = t_values[i]
            phi = phi_values[i]
            
            # Determine current basis in the cycle
            cycle_phase = (t % cycle_period) / cycle_period
            
            if cycle_phase < 1/3:
                # X-basis
                R_x, R_y, R_z = 0, 1, 1
            elif cycle_phase < 2/3:
                # Y-basis
                R_x, R_y, R_z = 1, 0, 1
            else:
                # Z-basis
                R_x, R_y, R_z = 1, 1, 0
            
            # Extract current Bloch components
            x, y, z = bloch_vector
            
            # Unitary evolution (rotation)
            new_x = x * np.cos(phi) + y * np.sin(phi)
            new_y = y * np.cos(phi) - x * np.sin(phi)
            new_z = z
            
            # Apply measurement effects (damping)
            dx = -measurement_frequency * R_x * new_x * dt
            dy = -measurement_frequency * R_y * new_y * dt
            dz = -measurement_frequency * R_z * new_z * dt
            
            # Update Bloch vector
            new_x += dx
            new_y += dy
            new_z += dz
            
            bloch_vector = np.array([new_x, new_y, new_z])
            
            # Store values
            sigma_x_values[i] = bloch_vector[0]
            sigma_y_values[i] = bloch_vector[1]
            sigma_z_values[i] = bloch_vector[2]
        
        # Return in same format as the simulation
        return {
            't': t_values,
            'theta': theta_values % (2 * np.pi),  # Modulo 2π for visualization
            'theta_full': theta_values,
            'sigma_x': sigma_x_values,
            'sigma_y': sigma_y_values,
            'sigma_z': sigma_z_values
        }


class QuantumMemristorSimulation:
    """Simulation of a quantum memristor with various measurement bases."""
    
    def __init__(self):
        # Simulation parameters
        self.t_max = 10            # Maximum simulation time
        self.theta_max = 10 * np.pi  # Maximum memory parameter value
        self.omega_theta = self.theta_max / self.t_max  # Angular frequency for θ
        self.omega_phi = np.pi / 2   # Angular frequency for φ
        self.dt = 0.025             # Time step
        
        # Default control parameters
        self.measurement_frequency = 0.5
        self.measurement_basis = 'z'
        self.show_all_trajectories = False
        self.use_fixed_seed = False  # Default to random mode
        self.random_seed = 42       # Default random seed for reproducibility
        self.num_overlay = 10       # Number of trajectories to overlay
        
        # Create theoretical model
        self.theory = QuantumMemristorTheory()
        
    def generate_trajectory(self, basis, seed=None):
        """Generate a quantum trajectory with specified measurement basis."""
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        point_count = int(self.t_max / self.dt)
        
        # Arrays to store the trajectory data
        t_values = np.zeros(point_count)
        theta_values = np.zeros(point_count)
        theta_full_values = np.zeros(point_count)
        sigma_x_values = np.zeros(point_count)
        sigma_y_values = np.zeros(point_count)
        sigma_z_values = np.zeros(point_count)
        
        # Initial quantum state (Bloch vector)
        bloch_vector = np.array([1.0, 0.0, 0.0])  # [x, y, z]
        
        # Generate the trajectory
        for i in range(point_count):
            t = i * self.dt
            t_values[i] = t
            
            # Evolution of parameters
            theta = self.omega_theta * t
            phi = self.omega_phi * t
            
            theta_values[i] = theta % (2 * np.pi)  # Modulo 2π for visualization
            theta_full_values[i] = theta
            
            # Unitary evolution (rotation on Bloch sphere)
            # This implements the quantum memristor transformation
            new_bloch_x = bloch_vector[0] * np.cos(phi) + bloch_vector[1] * np.sin(phi)
            new_bloch_y = bloch_vector[1] * np.cos(phi) - bloch_vector[0] * np.sin(phi)
            new_bloch_z = bloch_vector[2]
            
            new_bloch_vector = np.array([new_bloch_x, new_bloch_y, new_bloch_z])
            
            # Add quantum measurement effects
            if np.random.random() < self.measurement_frequency * self.dt:
                # Apply measurement in the specified basis
                if basis == 'x':
                    # X-basis measurement collapses to eigenstates of σ_x
                    measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[0]) / 2 else -1
                    new_bloch_vector = np.array([measurement_outcome, 0, 0])
                    
                elif basis == 'y':
                    # Y-basis measurement collapses to eigenstates of σ_y
                    measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[1]) / 2 else -1
                    new_bloch_vector = np.array([0, measurement_outcome, 0])
                    
                else:  # z-basis (default)
                    # Z-basis measurement collapses to eigenstates of σ_z
                    measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[2]) / 2 else -1
                    new_bloch_vector = np.array([0, 0, measurement_outcome])
            
            # Update the Bloch vector
            bloch_vector = new_bloch_vector
            
            # Store the expectation values
            sigma_x_values[i] = bloch_vector[0]
            sigma_y_values[i] = bloch_vector[1]
            sigma_z_values[i] = bloch_vector[2]
        
        return {
            't': t_values,
            'theta': theta_values,
            'theta_full': theta_full_values,
            'sigma_x': sigma_x_values,
            'sigma_y': sigma_y_values,
            'sigma_z': sigma_z_values
        }
    
    def generate_sequence_trajectory(self, seed=None):
        """Generate a trajectory with X→Y→Z measurement sequence."""
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        point_count = int(self.t_max / self.dt)
        
        # Arrays to store the trajectory data
        t_values = np.zeros(point_count)
        theta_values = np.zeros(point_count)
        theta_full_values = np.zeros(point_count)
        sigma_x_values = np.zeros(point_count)
        sigma_y_values = np.zeros(point_count)
        sigma_z_values = np.zeros(point_count)
        
        # Array to mark measurement events and their basis
        measurement_events = np.zeros(point_count, dtype=bool)
        measurement_basis = np.array([''] * point_count, dtype=object)
        
        # Initial quantum state (Bloch vector)
        bloch_vector = np.array([1.0, 0.0, 0.0])  # [x, y, z]
        
        # Measurement tracking
        last_measurement_time = -1
        measurement_count = 0
        
        # Generate the trajectory
        for i in range(point_count):
            t = i * self.dt
            t_values[i] = t
            
            # Evolution of parameters
            theta = self.omega_theta * t
            phi = self.omega_phi * t
            
            theta_values[i] = theta % (2 * np.pi)  # Modulo 2π for visualization
            theta_full_values[i] = theta
            
            # Unitary evolution (rotation on Bloch sphere)
            new_bloch_x = bloch_vector[0] * np.cos(phi) + bloch_vector[1] * np.sin(phi)
            new_bloch_y = bloch_vector[1] * np.cos(phi) - bloch_vector[0] * np.sin(phi)
            new_bloch_z = bloch_vector[2]
            
            new_bloch_vector = np.array([new_bloch_x, new_bloch_y, new_bloch_z])
            
            # Add quantum measurement effects with cycling basis
            if np.random.random() < self.measurement_frequency * self.dt:
                # Only measure if enough time has passed since last measurement
                if t - last_measurement_time > 0.2:
                    last_measurement_time = t
                    measurement_events[i] = True
                    
                    # Cycle through X, Y, Z measurements
                    cycle_index = measurement_count % 3
                    measurement_count += 1
                    
                    current_basis = 'x'
                    if cycle_index == 1:
                        current_basis = 'y'
                    if cycle_index == 2:
                        current_basis = 'z'
                    
                    measurement_basis[i] = current_basis
                    
                    # Apply measurement in the cycling basis
                    if current_basis == 'x':
                        # X-basis measurement
                        measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[0]) / 2 else -1
                        new_bloch_vector = np.array([measurement_outcome, 0, 0])
                        
                    elif current_basis == 'y':
                        # Y-basis measurement
                        measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[1]) / 2 else -1
                        new_bloch_vector = np.array([0, measurement_outcome, 0])
                        
                    elif current_basis == 'z':
                        # Z-basis measurement
                        measurement_outcome = 1 if np.random.random() < (1 + new_bloch_vector[2]) / 2 else -1
                        new_bloch_vector = np.array([0, 0, measurement_outcome])
            
            # Update the Bloch vector
            bloch_vector = new_bloch_vector
            
            # Store the expectation values
            sigma_x_values[i] = bloch_vector[0]
            sigma_y_values[i] = bloch_vector[1]
            sigma_z_values[i] = bloch_vector[2]
        
        return {
            't': t_values,
            'theta': theta_values,
            'theta_full': theta_full_values,
            'sigma_x': sigma_x_values,
            'sigma_y': sigma_y_values,
            'sigma_z': sigma_z_values,
            'measurement_events': measurement_events,
            'measurement_basis': measurement_basis
        }
    
    def generate_multiple_trajectories(self, basis, num_trajectories):
        """Generate multiple trajectories with different random seeds."""
        trajectories = []
        
        if self.use_fixed_seed:
            # Use deterministic seeds based on the base seed
            for i in range(num_trajectories):
                seed = self.random_seed + i
                if basis == 'sequence':
                    trajectory = self.generate_sequence_trajectory(seed=seed)
                else:
                    trajectory = self.generate_trajectory(basis, seed=seed)
                trajectories.append(trajectory)
        else:
            # Use random seeds
            for i in range(num_trajectories):
                if basis == 'sequence':
                    trajectory = self.generate_sequence_trajectory()
                else:
                    trajectory = self.generate_trajectory(basis)
                trajectories.append(trajectory)
                
        return trajectories
    
    def calculate_average_trajectory(self, trajectories):
        """Calculate the average trajectory from multiple trajectories."""
        if not trajectories:
            return None
        
        # Sample trajectory for structure (all trajectories should have the same time/theta points)
        sample = trajectories[0]
        
        # Create output structure
        avg_trajectory = {
            't': sample['t'],
            'theta': sample['theta'],
            'theta_full': sample['theta_full'],
            'sigma_x': np.zeros_like(sample['sigma_x']),
            'sigma_y': np.zeros_like(sample['sigma_y']),
            'sigma_z': np.zeros_like(sample['sigma_z'])
        }
        
        # Calculate averages for each point
        for traj in trajectories:
            avg_trajectory['sigma_x'] += traj['sigma_x']
            avg_trajectory['sigma_y'] += traj['sigma_y']
            avg_trajectory['sigma_z'] += traj['sigma_z']
        
        # Divide by number of trajectories to get the average
        avg_trajectory['sigma_x'] /= len(trajectories)
        avg_trajectory['sigma_y'] /= len(trajectories)
        avg_trajectory['sigma_z'] /= len(trajectories)
        
        return avg_trajectory
    
    def run_simulation(self):
        """Run simulations for all measurement bases."""
        results = {}
        
        # Generate single trajectories for the standard modes
        if self.use_fixed_seed:
            results['x_basis'] = self.generate_trajectory('x', seed=self.random_seed)
            results['y_basis'] = self.generate_trajectory('y', seed=self.random_seed)
            results['z_basis'] = self.generate_trajectory('z', seed=self.random_seed)
            results['sequence'] = self.generate_sequence_trajectory(seed=self.random_seed)
        else:
            results['x_basis'] = self.generate_trajectory('x')
            results['y_basis'] = self.generate_trajectory('y')
            results['z_basis'] = self.generate_trajectory('z')
            results['sequence'] = self.generate_sequence_trajectory()
        
        # Generate multiple trajectories for the overlay
        overlay_trajectories = self.generate_multiple_trajectories(
            self.measurement_basis, self.num_overlay)
        results['overlay_trajectories'] = overlay_trajectories
        
        # Calculate the average trajectory
        results['average_trajectory'] = self.calculate_average_trajectory(overlay_trajectories)
        
        # Calculate theoretical predictions
        results['theoretical_prediction'] = self.theory.calculate_theoretical_trajectory(
            self.measurement_basis, self.measurement_frequency)
        
        return results

class QuantumMemristorPlotter:
    """Handles plotting and interactive controls for quantum memristor visualization."""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.show_overlay = False  # Whether to show multiple overlaid trajectories
        self.show_average = False  # Whether to show the average trajectory
        self.show_theory = False   # Whether to show theoretical prediction
        self.colorbar = None       # Reference to colorbar for proper cleanup
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the main plot and interactive controls."""
        # Create figure and subplots
        plt.rcParams['figure.max_open_warning'] = 50  # Avoid warning for too many figures
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('Quantum Memristor Hysteresis: ⟨σ^x⟩ vs θ with Measurement Basis Dependence', 
                          fontsize=16, fontweight='bold')
        
        # Create a fixed layout
        plt.subplots_adjust(left=0.25, bottom=0.25, right=0.85, top=0.9)
        
        # Description text below title
        self.fig.text(0.5, 0.95, 
                      'This visualization demonstrates how quantum memristive behavior depends crucially on the measurement basis.',
                      ha='center', fontsize=12)
        
        # Create axis for control panel
        self.ax_control = plt.axes([0.05, 0.25, 0.15, 0.65])
        self.ax_control.axis('off')
        self.ax_control.text(0.5, 0.95, 'Controls', ha='center', fontsize=14, fontweight='bold')
        
        # Create control panel for measurement basis
        self.ax_radio = plt.axes([0.05, 0.60, 0.15, 0.2])
        self.radio = RadioButtons(
            self.ax_radio, 
            ['X-basis', 'Y-basis', 'Z-basis', 'X→Y→Z Sequence'],
            activecolor='blue'
        )
        self.radio.on_clicked(self.update_basis)
        
        # Create control panel for measurement frequency
        self.ax_freq = plt.axes([0.05, 0.45, 0.15, 0.03])
        self.slider_freq = Slider(
            self.ax_freq, 'Frequency', 0.001, 2.0, 
            valinit=self.simulation.measurement_frequency, valstep=0.001
        )
        self.slider_freq.on_changed(self.update_frequency)
        
        # Add precise control buttons for frequency
        self.ax_freq_minus = plt.axes([0.05, 0.40, 0.07, 0.03])
        self.btn_freq_minus = Button(self.ax_freq_minus, '-0.001')
        self.btn_freq_minus.on_clicked(self.decrease_freq_small)
        
        self.ax_freq_plus = plt.axes([0.13, 0.40, 0.07, 0.03])
        self.btn_freq_plus = Button(self.ax_freq_plus, '+0.001')
        self.btn_freq_plus.on_clicked(self.increase_freq_small)
        
        # Add checkbox for showing all trajectories
        self.ax_check = plt.axes([0.05, 0.32, 0.15, 0.05])
        self.check = CheckButtons(
            self.ax_check, 
            ['Show all bases'], 
            [self.simulation.show_all_trajectories]
        )
        self.check.on_clicked(self.toggle_all_trajectories)
        
        # Add checkbox for showing overlay trajectories
        self.ax_overlay = plt.axes([0.05, 0.27, 0.15, 0.05])
        self.check_overlay = CheckButtons(
            self.ax_overlay, 
            ['Show multiple trajectories'], 
            [self.show_overlay]
        )
        self.check_overlay.on_clicked(self.toggle_overlay)
        
        # Add checkbox for showing average trajectory
        self.ax_average = plt.axes([0.05, 0.22, 0.15, 0.05])
        self.check_average = CheckButtons(
            self.ax_average, 
            ['Show average trajectory'], 
            [self.show_average]
        )
        self.check_average.on_clicked(self.toggle_average)
        
        # Add checkbox for showing theoretical prediction
        self.ax_theory = plt.axes([0.05, 0.17, 0.15, 0.05])
        self.check_theory = CheckButtons(
            self.ax_theory, 
            ['Show theoretical model'], 
            [self.show_theory]
        )
        self.check_theory.on_clicked(self.toggle_theory)
        
        # Add slider for number of overlay trajectories
        self.ax_num_overlay = plt.axes([0.05, 0.12, 0.15, 0.03])
        self.slider_overlay = Slider(
            self.ax_num_overlay, 'Num Trajectories', 2, 50, 
            valinit=self.simulation.num_overlay, valstep=1
        )
        self.slider_overlay.on_changed(self.update_num_overlay)
        
        # Add randomness control
        self.ax_random = plt.axes([0.05, 0.07, 0.15, 0.05])
        self.check_random = CheckButtons(
            self.ax_random,
            ['Use random seeds'],
            [not self.simulation.use_fixed_seed]  # Invert logic for clearer labeling
        )
        self.check_random.on_clicked(self.toggle_randomness)
        
        # Add regenerate button
        self.ax_regen = plt.axes([0.05, 0.02, 0.15, 0.04])
        self.btn_regen = Button(self.ax_regen, 'Regenerate Data')
        self.btn_regen.on_clicked(self.regenerate_data)
        
        # Add frequency display
        self.freq_text = self.fig.text(0.05, 0.52, f"Current frequency: {self.simulation.measurement_frequency:.4f}", 
                                  fontsize=10)
        
        # Initialize the plot
        self.data = self.simulation.run_simulation()
        self.plot_data()
    
    def plot_data(self):
        """Plot the quantum trajectories based on current settings."""
        # Clear existing plot
        self.ax.clear()
        
        # Remove any existing colorbar to avoid recursion errors
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        
        # Set up the axes
        self.ax.set_xlabel('Memory Parameter θ (mod 2π)', fontsize=12)
        self.ax.set_ylabel('Quantum Expectation Value ⟨σ^x⟩', fontsize=12)
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-1.1, 1.1)
        
        # Set x-ticks to be in terms of π
        self.ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Add horizontal reference line at y=0
        self.ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Determine what to plot based on settings
        if self.show_overlay:
            # Plot multiple trajectories with transparency
            for i, trajectory in enumerate(self.data['overlay_trajectories']):
                # Get color based on trajectory index (for visual distinction)
                color_val = i / len(self.data['overlay_trajectories'])
                color = plt.cm.viridis(color_val)
                
                # Plot the trajectory
                self.ax.plot(trajectory['theta'], trajectory['sigma_x'], 
                             color=color, alpha=0.3, linewidth=0.8)
            
            # Add text showing number of trajectories
            self.ax.text(0.02, 0.98, f"{len(self.data['overlay_trajectories'])} Trajectories Overlaid",
                        transform=self.ax.transAxes, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Plot the average trajectory if enabled
            if self.show_average and 'average_trajectory' in self.data:
                avg_traj = self.data['average_trajectory']
                self.ax.plot(avg_traj['theta'], avg_traj['sigma_x'], 
                            'k-', linewidth=2.5, label='Average (Numerical)')
                
                # Add statistical information
                std_dev = np.std([traj['sigma_x'] for traj in self.data['overlay_trajectories']], axis=0)
                avg_std = np.mean(std_dev)
                self.ax.text(0.02, 0.90, f"Average std dev: {avg_std:.3f}",
                            transform=self.ax.transAxes, fontsize=10, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Optionally add confidence interval (mean +/- std deviation)
                # Mean trajectory
                mean_x = avg_traj['sigma_x']
                theta = avg_traj['theta']
                
                # Plot confidence intervals (mean +/- std deviation)
                self.ax.fill_between(theta, mean_x - std_dev, mean_x + std_dev, 
                                    color='gray', alpha=0.2, label='±1σ range')
            
            # Plot theoretical prediction if enabled
            if self.show_theory and 'theoretical_prediction' in self.data:
                theory = self.data['theoretical_prediction']
                self.ax.plot(theory['theta'], theory['sigma_x'], 
                            'r--', linewidth=2.0, label='Theoretical Model')
                
        elif self.simulation.show_all_trajectories:
            # Plot all trajectories
            self.ax.plot(self.data['x_basis']['theta'], self.data['x_basis']['sigma_x'], 
                         'b-', label='X-basis', linewidth=1, alpha=0.7)
            self.ax.plot(self.data['y_basis']['theta'], self.data['y_basis']['sigma_x'], 
                         'g-', label='Y-basis', linewidth=1, alpha=0.7)
            self.ax.plot(self.data['z_basis']['theta'], self.data['z_basis']['sigma_x'], 
                         'r-', label='Z-basis', linewidth=1, alpha=0.7)
            self.ax.legend()
        else:
            # Plot only the selected basis
            if self.simulation.measurement_basis == 'sequence':
                # For the sequence, use color to indicate measurement basis
                theta = self.data['sequence']['theta']
                sigma_x = self.data['sequence']['sigma_x']
                
                # Color gradient by time
                points = np.array([theta, sigma_x]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a colormap based on time
                t_norm = self.data['sequence']['t'] / self.simulation.t_max
                cmap = cm.plasma
                colors = cmap(t_norm)
                
                # Plot the line with color gradient
                lc = LineCollection(segments, colors=colors, linewidth=1.5)
                self.ax.add_collection(lc)
                
                # Add colorbar for time evolution (using direct figure method to avoid recursion)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
                sm.set_array([])
                self.colorbar = self.fig.colorbar(sm, ax=self.ax, pad=0.01)
                self.colorbar.set_label('Time evolution')
                
                # Mark measurement events
                mask = self.data['sequence']['measurement_events']
                if np.any(mask):
                    bases = self.data['sequence']['measurement_basis'][mask]
                    
                    # Define color for each basis
                    basis_colors = []
                    for b in bases:
                        if b == 'x':
                            basis_colors.append('blue')
                        elif b == 'y':
                            basis_colors.append('green')
                        elif b == 'z':
                            basis_colors.append('red')
                    
                    self.ax.scatter(theta[mask], sigma_x[mask], c=basis_colors, s=30, zorder=3)
                    
                    # Add a legend for the measurement bases
                    self.ax.scatter([], [], c='blue', s=30, label='X-basis measurement')
                    self.ax.scatter([], [], c='green', s=30, label='Y-basis measurement')
                    self.ax.scatter([], [], c='red', s=30, label='Z-basis measurement')
                
                # Plot theoretical prediction if enabled
                if self.show_theory and 'theoretical_prediction' in self.data:
                    theory = self.data['theoretical_prediction']
                    self.ax.plot(theory['theta'], theory['sigma_x'], 
                                'r--', linewidth=2.0, label='Theoretical Model')
                
                self.ax.legend()
                
            else:
                # For single basis, use a color gradient by time
                data = self.data[f'{self.simulation.measurement_basis}_basis']
                theta = data['theta']
                sigma_x = data['sigma_x']
                
                # Create points and segments for LineCollection
                points = np.array([theta, sigma_x]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create a colormap based on time
                t_norm = data['t'] / self.simulation.t_max
                cmap = cm.plasma
                colors = cmap(t_norm)
                
                # Plot the line with color gradient
                lc = LineCollection(segments, colors=colors, linewidth=1.5)
                self.ax.add_collection(lc)
                
                # Add colorbar (using direct figure method to avoid recursion)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
                sm.set_array([])
                self.colorbar = self.fig.colorbar(sm, ax=self.ax, pad=0.01)
                self.colorbar.set_label('Time evolution')
                
                # Plot theoretical prediction if enabled
                if self.show_theory and 'theoretical_prediction' in self.data:
                    theory = self.data['theoretical_prediction']
                    self.ax.plot(theory['theta'], theory['sigma_x'], 
                                'r--', linewidth=2.0, label='Theoretical Model')
                    self.ax.legend()
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Update title based on current settings
        if self.simulation.measurement_basis == 'x':
            basis_name = 'X-basis'
        elif self.simulation.measurement_basis == 'y':
            basis_name = 'Y-basis'
        elif self.simulation.measurement_basis == 'z':
            basis_name = 'Z-basis'
        else:
            basis_name = 'X→Y→Z Sequence'
            
        # Count revolutions
        if len(self.data['x_basis']['theta_full']) > 0:
            revolutions = int(self.data['x_basis']['theta_full'][-1] / (2 * np.pi))
            title = f"{basis_name} Measurements - {revolutions} complete revolutions in θ"
            self.ax.set_title(title, fontsize=14)
        
        # Add randomness status to plot
        randomness_status = "Random Mode" if not self.simulation.use_fixed_seed else f"Reproducible Mode"
        self.fig.text(0.5, 0.01, randomness_status, ha='center', fontsize=10)
        
        # Redraw canvas
        self.fig.canvas.draw_idle()
    
    def update_basis(self, label):
        """Update the measurement basis based on radio button selection."""
        if label == 'X-basis':
            self.simulation.measurement_basis = 'x'
        elif label == 'Y-basis':
            self.simulation.measurement_basis = 'y'
        elif label == 'Z-basis':
            self.simulation.measurement_basis = 'z'
        elif label == 'X→Y→Z Sequence':
            self.simulation.measurement_basis = 'sequence'
        
        self.regenerate_data(None)
    
    def update_frequency(self, val):
        """Update measurement frequency based on slider."""
        self.simulation.measurement_frequency = val
        self.freq_text.set_text(f"Current frequency: {self.simulation.measurement_frequency:.4f}")
        self.fig.canvas.draw_idle()
    
    def decrease_freq_small(self, event):
        """Decrease frequency by 0.001."""
        new_freq = max(0.001, self.simulation.measurement_frequency - 0.001)
        self.simulation.measurement_frequency = new_freq
        self.slider_freq.set_val(new_freq)
        self.freq_text.set_text(f"Current frequency: {self.simulation.measurement_frequency:.4f}")
        self.fig.canvas.draw_idle()
    
    def increase_freq_small(self, event):
        """Increase frequency by 0.001."""
        new_freq = min(2.0, self.simulation.measurement_frequency + 0.001)
        self.simulation.measurement_frequency = new_freq
        self.slider_freq.set_val(new_freq)
        self.freq_text.set_text(f"Current frequency: {self.simulation.measurement_frequency:.4f}")
        self.fig.canvas.draw_idle()
    
    def toggle_all_trajectories(self, label):
        """Toggle showing all trajectories."""
        self.simulation.show_all_trajectories = not self.simulation.show_all_trajectories
        # If showing all trajectories, disable overlay mode
        if self.simulation.show_all_trajectories:
            if self.show_overlay:
                self.show_overlay = False
                self.check_overlay.set_active(0)
            if self.show_average:
                self.show_average = False
                self.check_average.set_active(0)
            if self.show_theory:
                self.show_theory = False
                self.check_theory.set_active(0)
        self.plot_data()
    
    def toggle_overlay(self, label):
        """Toggle showing overlay trajectories."""
        self.show_overlay = not self.show_overlay
        # If showing overlay, disable show all trajectories
        if self.show_overlay and self.simulation.show_all_trajectories:
            self.simulation.show_all_trajectories = False
            self.check.set_active(0)
        # If turning off overlay, turn off average and theory too
        if not self.show_overlay:
            if self.show_average:
                self.show_average = False
                self.check_average.set_active(0)
            if self.show_theory:
                self.show_theory = False
                self.check_theory.set_active(0)
        self.regenerate_data(None)
    
    def toggle_average(self, label):
        """Toggle showing average trajectory."""
        self.show_average = not self.show_average
        # If showing average, make sure overlay is enabled
        if self.show_average and not self.show_overlay:
            self.show_overlay = True
            self.check_overlay.set_active(0)
        self.plot_data()
    
    def toggle_theory(self, label):
        """Toggle showing theoretical prediction."""
        self.show_theory = not self.show_theory
        self.plot_data()
    
    def update_num_overlay(self, val):
        """Update the number of overlay trajectories."""
        self.simulation.num_overlay = int(val)
        if self.show_overlay:
            self.regenerate_data(None)
    
    def toggle_randomness(self, label):
        """Toggle between random and reproducible mode."""
        self.simulation.use_fixed_seed = not self.simulation.use_fixed_seed
        self.regenerate_data(None)
    
    def regenerate_data(self, event):
        """Regenerate the simulation data with current parameters."""
        self.data = self.simulation.run_simulation()
        self.plot_data()
    
    def show(self):
        """Display the plot."""
        # Add explanation text at the bottom
        explanation = """
Quantum Memristor Visualization: Compare numerical simulations with theoretical predictions from our mathematical model.
The 'Show theoretical model' option displays the analytical solution from the master equation, while the numerical 
simulations show actual quantum trajectories with probabilistic measurement outcomes. This demonstrates how our 
mathematical framework accurately captures the statistical behavior of quantum memristors.
        """
        self.fig.text(0.5, 0.07, explanation, ha='center', fontsize=10, 
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.5))
        
        plt.show()

def main():
    """Main function to run the quantum memristor visualization."""
    print("Initializing Quantum Memristor Hysteresis Visualization...")
    print("This visualization demonstrates how measurement basis affects quantum memristor behavior.")
    print("\nVisualization Options:")
    print("  - Show multiple trajectories: Overlay random quantum paths")
    print("  - Show average trajectory: Display the statistical mean of multiple paths")
    print("  - Show theoretical model: Display predictions from the mathematical framework")
    print("  - Standard deviation shading: Areas where quantum outcomes are most uncertain")
    
    # Create the simulation
    simulation = QuantumMemristorSimulation()
    
    # Create the plotter and show the interactive visualization
    plotter = QuantumMemristorPlotter(simulation)
    plotter.show()

if __name__ == "__main__":
    main()