"""
Network Visualization Script
Displays the structure of the self-modifying network with ASCII art and PNG diagram
"""

import pickle
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from datetime import datetime


# Custom unpickler to handle module path changes
class ModuleRenameUnpickler(pickle.Unpickler):
    """Unpickler that remaps old module names to new classes. prefix"""
    def find_class(self, module, name):
        # Remap old direct imports to classes. package
        if module in ['network', 'layer', 'neuron', 'reward', 'backward', 
                      'modification_tracker', 'meta_learner']:
            module = f'classes.{module}'
        # Also handle old 'class.' prefix
        elif module.startswith('class.'):
            module = module.replace('class.', 'classes.')
        # Handle gpu_config moved to setup folder
        elif module == 'gpu_config':
            module = 'setup.gpu_config'
        return super().find_class(module, name)


def load_network():
    """Load the saved network checkpoint"""
    # Try both possible checkpoint locations
    checkpoint_paths = [
        Path("checkpoints/network_state.pkl"),
        Path("network_checkpoint.pkl"),
        Path("network_state.pkl")
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if path.exists():
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("❌ No checkpoint found. Train the network first with:")
        print("   python continuous_train.py --new")
        return None
    
    print(f"📂 Loading from: {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            # Use custom unpickler to handle module path changes
            unpickler = ModuleRenameUnpickler(f)
            data = unpickler.load()
        
        # Check if it's a dict (checkpoint format) or direct network object
        if isinstance(data, dict):
            if 'network' in data:
                network = data['network']
            elif 'state' in data:
                network = data['state']
            else:
                # Assume the dict IS the network state
                print(f"⚠️  Checkpoint is a dict with keys: {list(data.keys())}")
                print(f"⚠️  Cannot extract network object. Try re-saving with continuous_train.py")
                return None
        else:
            network = data
        
        return network
    except Exception as e:
        print(f"❌ Error loading network: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_connection_strength(layer_idx, neuron_idx, next_neuron_idx, network):
    """Get average connection strength between two neurons"""
    if layer_idx >= len(network.layers) - 1:
        return 0.0
    
    current_layer = network.layers[layer_idx]
    if neuron_idx >= len(current_layer.neurons):
        return 0.0
    
    neuron = current_layer.neurons[neuron_idx]
    
    # Get weight for this specific output neuron
    if next_neuron_idx < len(neuron.weights):
        return abs(neuron.weights[next_neuron_idx])
    return 0.0


def visualize_network(network):
    """Display network structure with ASCII art"""
    
    print("\n" + "="*80)
    print("NEURAL NETWORK STRUCTURE VISUALIZATION")
    print("="*80)
    
    # Network summary
    print(f"\n📊 Network Summary:")
    print(f"   Total Layers: {len(network.layers)}")
    print(f"   Total Neurons: {sum(len(layer.neurons) for layer in network.layers)}")
    print(f"   Training Steps: {network.training_steps}")
    print(f"   Recent Avg Reward: {network.reward_function.get_average_reward():.2f}")
    print(f"   Recent Avg Error: {network.reward_function.get_average_error():.2f}")
    
    # Layer details
    print(f"\n📐 Layer Architecture:")
    for i, layer in enumerate(network.layers):
        layer_type = "Input" if i == 0 else "Output" if i == len(network.layers)-1 else "Hidden"
        print(f"   Layer {i} ({layer_type}): {len(layer.neurons)} neurons")
    
    # Visual representation
    print(f"\n🔗 Network Connections:")
    print()
    
    # Calculate max neurons for spacing
    max_neurons = max(len(layer.neurons) for layer in network.layers)
    
    # Draw each layer
    for layer_idx, layer in enumerate(network.layers):
        neurons = layer.neurons
        n_neurons = len(neurons)
        
        # Layer label
        layer_type = "INPUT" if layer_idx == 0 else "OUTPUT" if layer_idx == len(network.layers)-1 else f"HIDDEN"
        print(f"  Layer {layer_idx} [{layer_type}]")
        print()
        
        # Draw neurons
        for neuron_idx, neuron in enumerate(neurons):
            # Neuron representation with performance indicator
            perf = neuron.get_average_performance()
            
            # Performance color/symbol
            if perf > 0.5:
                symbol = "●"  # Good performance
                perf_str = "✓"
            elif perf > -0.5:
                symbol = "◐"  # Medium performance
                perf_str = "~"
            else:
                symbol = "○"  # Poor performance
                perf_str = "✗"
            
            # Display neuron
            activation_count = neuron.activation_count
            print(f"    {symbol} N{neuron_idx} [{perf_str}] (perf={perf:+.2f}, activations={activation_count})")
            
            # Show connections to next layer
            if layer_idx < len(network.layers) - 1:
                next_layer = network.layers[layer_idx + 1]
                n_next = len(next_layer.neurons)
                
                # Get weight matrix - neuron.weights is shape (input_dim, output_dim)
                weight_matrix = neuron.weights
                
                # Calculate connection strengths (average absolute weight to each output neuron)
                strong_conns = []
                medium_conns = []
                weak_conns = []
                
                for next_idx in range(min(n_next, weight_matrix.shape[1])):
                    # Get all weights connecting to this output neuron
                    output_weights = weight_matrix[:, next_idx]
                    # Use mean absolute weight as connection strength
                    avg_abs_weight = np.mean(np.abs(output_weights))
                    
                    if avg_abs_weight > 5.0:
                        strong_conns.append((next_idx, avg_abs_weight))
                    elif avg_abs_weight > 1.0:
                        medium_conns.append((next_idx, avg_abs_weight))
                    else:
                        weak_conns.append((next_idx, avg_abs_weight))
                
                # Display connections
                if strong_conns:
                    conn_str = ", ".join([f"N{idx}({w:+.1f})" for idx, w in strong_conns[:3]])
                    if len(strong_conns) > 3:
                        conn_str += f" +{len(strong_conns)-3} more"
                    print(f"      ══► Strong: {conn_str}")
                
                if medium_conns and len(strong_conns) < 3:
                    conn_str = ", ".join([f"N{idx}({w:+.1f})" for idx, w in medium_conns[:2]])
                    if len(medium_conns) > 2:
                        conn_str += f" +{len(medium_conns)-2} more"
                    print(f"      ──► Medium: {conn_str}")
                
                # Show total connection count
                total_conns = weight_matrix.shape[1]  # Number of output neurons
                print(f"      └── {total_conns} total connections to Layer {layer_idx+1}")
            
            print()
        
        # Draw separator between layers
        if layer_idx < len(network.layers) - 1:
            print("      ║")
            print("      ╚═══════════════════════════════════════════════")
            print("      ║")
            print()
    
    # Connection matrix summary
    print(f"\n📊 Connection Matrix Summary:")
    for layer_idx in range(len(network.layers) - 1):
        current_layer = network.layers[layer_idx]
        next_layer = network.layers[layer_idx + 1]
        
        print(f"\n   Layer {layer_idx} → Layer {layer_idx+1}:")
        print(f"   {len(current_layer.neurons)} neurons → {len(next_layer.neurons)} neurons")
        
        # Calculate statistics - neuron.weights is a 2D matrix (input_dim, output_dim)
        all_weights = []
        for neuron in current_layer.neurons:
            # Flatten the weight matrix to get all individual weights
            all_weights.extend(neuron.weights.flatten())
        
        if all_weights:
            all_weights = np.array(all_weights)
            print(f"   Weight stats: mean={np.mean(all_weights):.3f}, "
                  f"std={np.std(all_weights):.3f}, "
                  f"range=[{np.min(all_weights):.3f}, {np.max(all_weights):.3f}]")
            print(f"   Strong connections (|w|>5): {np.sum(np.abs(all_weights) > 5.0)}/{len(all_weights)}")
    
    # Neuron performance summary
    print(f"\n🎯 Neuron Performance Summary:")
    for layer_idx, layer in enumerate(network.layers):
        performances = [n.get_average_performance() for n in layer.neurons]
        if performances:
            print(f"   Layer {layer_idx}: avg={np.mean(performances):+.3f}, "
                  f"range=[{np.min(performances):+.3f}, {np.max(performances):+.3f}]")


def visualize_compact(network):
    """Compact visualization showing network topology"""
    print("\n" + "="*80)
    print("COMPACT NETWORK TOPOLOGY")
    print("="*80 + "\n")
    
    # Build ASCII diagram
    layers_info = []
    for i, layer in enumerate(network.layers):
        n = len(layer.neurons)
        layers_info.append(n)
    
    # Draw input
    print("  INPUT (4D)")
    print("  [num1, num2, op1, op2]")
    print("       │")
    print("       ▼")
    
    # Draw layers
    for i, n_neurons in enumerate(layers_info):
        layer_type = "HIDDEN" if i < len(layers_info)-1 else "OUTPUT"
        
        # Draw neurons
        print(f"\n  ┌─────────────────┐")
        print(f"  │ Layer {i} ({layer_type})  │")
        print(f"  │  {n_neurons} neurons       │")
        print(f"  └─────────────────┘")
        
        # Draw connections to next layer
        if i < len(layers_info) - 1:
            next_n = layers_info[i+1]
            connections = n_neurons * next_n
            
            # Connection density visualization
            if connections > 100:
                conn_viz = "═" * 10
            elif connections > 50:
                conn_viz = "═" * 7 + "─" * 3
            elif connections > 20:
                conn_viz = "═" * 5 + "─" * 5
            else:
                conn_viz = "─" * 10
            
            print(f"       {conn_viz}▶ {connections} connections")
    
    # Draw output
    print("\n       │")
    print("       ▼")
    print("  PREDICTION (1D)")
    print("  [result]")
    print()


def show_legend():
    """Display legend for symbols"""
    print("\n" + "="*80)
    print("LEGEND")
    print("="*80)
    print("\nNeuron Performance:")
    print("  ● [✓] = Good performance (>0.5)")
    print("  ◐ [~] = Medium performance (-0.5 to 0.5)")
    print("  ○ [✗] = Poor performance (<-0.5)")
    print("\nConnection Strength:")
    print("  ══► Strong (|weight| > 5.0)")
    print("  ──► Medium (|weight| > 1.0)")
    print("  ··► Weak (|weight| ≤ 1.0)")
    print()


def generate_network_image(network, output_path="output/network_visualization.png"):
    """Generate a PNG image of the network with connections"""
    print(f"\n🎨 Generating network visualization image...")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Calculate layer positions
    n_layers = len(network.layers)
    layer_spacing = 1.0 / (n_layers + 1)
    
    # Store neuron positions for drawing connections
    neuron_positions = {}
    
    # Calculate max neurons for vertical spacing
    max_neurons = max(len(layer.neurons) for layer in network.layers)
    
    # Draw neurons and store positions
    for layer_idx, layer in enumerate(network.layers):
        n_neurons = len(layer.neurons)
        x = layer_spacing * (layer_idx + 1)
        
        # Vertical spacing for this layer
        if n_neurons == 1:
            y_positions = [0.5]
        else:
            y_spacing = 0.8 / max(n_neurons - 1, 1)
            y_start = 0.5 - (n_neurons - 1) * y_spacing / 2
            y_positions = [y_start + i * y_spacing for i in range(n_neurons)]
        
        # Draw each neuron
        for neuron_idx, neuron in enumerate(layer.neurons):
            y = y_positions[neuron_idx]
            neuron_positions[(layer_idx, neuron_idx)] = (x, y)
            
            # Color based on performance
            perf = neuron.get_average_performance()
            if perf > 0.5:
                color = '#2ecc71'  # Green
                edge_color = '#27ae60'
            elif perf > -0.5:
                color = '#f39c12'  # Orange
                edge_color = '#e67e22'
            else:
                color = '#e74c3c'  # Red
                edge_color = '#c0392b'
            
            # Draw neuron as circle
            circle = plt.Circle((x, y), 0.015, color=color, ec=edge_color, 
                              linewidth=2, zorder=3)
            ax.add_patch(circle)
            
            # Add neuron label
            ax.text(x, y, f'N{neuron_idx}', ha='center', va='center',
                   fontsize=6, fontweight='bold', color='white', zorder=4)
    
    # Draw connections between layers
    for layer_idx in range(len(network.layers) - 1):
        current_layer = network.layers[layer_idx]
        next_layer = network.layers[layer_idx + 1]
        
        for neuron_idx, neuron in enumerate(current_layer.neurons):
            x1, y1 = neuron_positions[(layer_idx, neuron_idx)]
            
            # Get weight matrix
            weight_matrix = neuron.weights
            
            # Draw connections to each neuron in next layer
            for next_idx in range(min(len(next_layer.neurons), weight_matrix.shape[1])):
                if (layer_idx + 1, next_idx) not in neuron_positions:
                    continue
                    
                x2, y2 = neuron_positions[(layer_idx + 1, next_idx)]
                
                # Calculate connection strength
                output_weights = weight_matrix[:, next_idx]
                avg_abs_weight = np.mean(np.abs(output_weights))
                
                # Determine line style and color based on weight
                if avg_abs_weight > 5.0:
                    alpha = 0.7
                    linewidth = 2.5
                    color = '#3498db'  # Strong blue
                elif avg_abs_weight > 1.0:
                    alpha = 0.4
                    linewidth = 1.5
                    color = '#95a5a6'  # Medium gray
                else:
                    alpha = 0.15
                    linewidth = 0.5
                    color = '#bdc3c7'  # Weak light gray
                
                # Draw connection line
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                       linewidth=linewidth, zorder=1)
    
    # Add layer labels
    for layer_idx in range(n_layers):
        x = layer_spacing * (layer_idx + 1)
        
        if layer_idx == 0:
            layer_name = "INPUT"
        elif layer_idx == n_layers - 1:
            layer_name = "OUTPUT"
        else:
            layer_name = f"HIDDEN {layer_idx}"
        
        n_neurons = len(network.layers[layer_idx].neurons)
        label = f"{layer_name}\n({n_neurons} neurons)"
        
        ax.text(x, 0.95, label, ha='center', va='top',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add title with network stats
    title = f"Self-Modifying Network Visualization\n"
    title += f"Training Steps: {network.training_steps} | "
    title += f"Avg Reward: {network.reward_function.get_average_reward():.2f} | "
    title += f"Avg Error: {network.reward_function.get_average_error():.2f}"
    ax.text(0.5, 1.0, title, ha='center', va='top',
           fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#2ecc71', lw=4, label='Good Performance (>0.5)'),
        plt.Line2D([0], [0], color='#f39c12', lw=4, label='Medium Performance (-0.5 to 0.5)'),
        plt.Line2D([0], [0], color='#e74c3c', lw=4, label='Poor Performance (<-0.5)'),
        plt.Line2D([0], [0], color='#3498db', lw=2.5, alpha=0.7, label='Strong Connection (|w|>5)'),
        plt.Line2D([0], [0], color='#95a5a6', lw=1.5, alpha=0.4, label='Medium Connection (|w|>1)'),
        plt.Line2D([0], [0], color='#bdc3c7', lw=0.5, alpha=0.15, label='Weak Connection (|w|≤1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.02),
             ncol=3, framealpha=0.9, fontsize=8)
    
    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', va='bottom',
           fontsize=7, style='italic', transform=ax.transAxes, alpha=0.6)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Network visualization saved to: {output_path}")


def main():
    """Main visualization function"""
    print("\n🔍 Loading Neural Network...")
    
    network = load_network()
    if network is None:
        return
    
    print("✓ Network loaded successfully!")
    
    # Generate PNG image
    generate_network_image(network)
    
    # Show compact topology
    visualize_compact(network)
    
    # Show detailed structure
    visualize_network(network)
    
    # Show legend
    show_legend()
    
    print("="*80)
    print()


if __name__ == "__main__":
    main()
