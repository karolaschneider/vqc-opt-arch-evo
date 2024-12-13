"""Variational Quantum Circuit"""

""" 
create agents that use quantum circuits 
to process observations and select actions in a 
reinforcement learning environment
"""

import numpy as np
import pennylane as qml
import torch
from torch import Tensor, nn

from typing import Optional, List

class GatePos:
    """Information about a gate in the circuit"""
    def __init__(self, gate, target, control=None):
        self.gate = gate
        self.target = target
        self.control = control

    def __repr__(self):
        control = '' if self.control is None else f',{self.control}'
        return f"{self.gate}({self.target}{control})"
    
    def get(self, key):
        return getattr(self, key, None)
    
    def __getitem__(self, key):
        if key in ['gate', 'target', 'control']:
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in GatePos")
    

def count_gates(gate_list: List[GatePos], types: List[str]):
    """Count the number of gates in the circuit"""
    count = 0
    for gate in gate_list:
        if gate.gate in types:
            count += 1
    return count


# VQC circuit Gate-Based
class VQCGate(nn.Module):
    """Variational quantum circuit"""

    # action space depends on env
    def __init__(self, num_qubits: int, gate_list: List[GatePos], action_space: int, circ_type: str, num_layers: Optional[int] = None):
        super().__init__()

        self.num_qubits = num_qubits

        # list representation
        self.gate_list: List[GatePos] = gate_list

        # action space dimension
        self.action_space = action_space

        # circuit type (gate or prototype)
        self.circ_type = circ_type
        
        self.num_layers = num_layers      

        # create qnode with choosen device
        self.device = qml.device("default.qubit", wires=range(num_qubits))
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        # parameters randomly initialized
        self.num_gates = len(gate_list)
        self.parameterized_gates = count_gates(gate_list, ['RX', 'RY', 'RZ'])
        self.weights = nn.Parameter(
            (torch.rand(size=(self.num_gates,)) * 2 * torch.pi) - torch.pi,
            requires_grad=False,
        )
        # bias randomly initialized
        self.bias = nn.Parameter((torch.rand(self.action_space) * 0.01), requires_grad=False)

    def circuit(self, weights: Tensor, input_data: Tensor, gate_list: List[GatePos]):
        """Construct and execute the quantum circuit"""
        qml.AmplitudeEmbedding(
            features=input_data, wires=range(self.num_qubits), pad_with=0.3, normalize=True
        )
        for i, gate in enumerate(gate_list):
            gate_name = gate.get('gate')
            if gate_name is None:
                raise ValueError("Gate name must be provided")
            if gate.get('control') is not None:
                getattr(qml, gate_name)(wires=[gate.get('control'), gate.get('target')])
            else:
                if gate.get('gate') in ['RX', 'RY', 'RZ']:
                    getattr(qml, gate_name)(phi=self.weights[i], wires=gate.get('target'))
                else:
                    getattr(qml, gate_name)(wires=gate.get('target'))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]

    def forward(self, input_data: Tensor, gate_list: List[GatePos]) -> Tensor:
        """Forward pass"""
        results = []
        for x_i in input_data:
            result = self.qnode(self.weights, x_i, gate_list)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            results.append(result)
        stacked_results = torch.stack(results)
        biased_results = stacked_results + self.bias
        return biased_results

    def show(self, input_data: Tensor, gate_list: List[GatePos]):
        """Prints the circuit"""
        draw = qml.draw(self.qnode)
        print(draw(self.weights, input_data, gate_list))

def gate_pos_to_array(gate_pos: GatePos):
    """Convert GatePos to numpy array"""
    return (gate_pos.gate, gate_pos.target, gate_pos.control)

def array_to_gate_pos(array):
    """Convert numpy array to GatePos"""
    return GatePos(gate=array['gate'], target=array['target'], control=array['control'])

# for testing pusposes
def create_SEL_circ(num_qubits: int, layer_count: int = 1) -> List[GatePos]:
    """Create a circuit for the SEL (strongly entangling layers) Ansatz"""
    circ = []
    for i in range (num_qubits - 1):
        circ.append(GatePos(gate="CNOT", target=i+1, control=i))
    circ.append(GatePos(gate="CNOT", target=0, control=num_qubits-1))
    for i in range (num_qubits):
        circ.append(GatePos(gate="RX", target=i))
        circ.append(GatePos(gate="RY", target=i))
        circ.append(GatePos(gate="RZ", target=i))
    circ = np.tile(circ, layer_count)
    circ = [array_to_gate_pos(gate) for gate in circ]
    return circ

def create_random_circ(num_gates: int, num_qubits: int, layer_count: int = 1, target_qubit: Optional[int] = None) -> List[GatePos]:
    """Create a random circuit"""
    gate_list = ["RX", "RY", "RZ", "CNOT"] # universal gate set
    circ = []
    if target_qubit is not None and target_qubit >= num_qubits:
        raise ValueError("Target qubit must be less than the number of qubits")
    for _ in range(num_gates):
        gate = np.random.choice(gate_list)
        target = target_qubit if target_qubit is not None else np.random.randint(num_qubits)
        if gate == "CNOT":
            control = np.random.choice([i for i in range(num_qubits) if i != target])
            circ.append(GatePos(gate=gate, target=target, control=control))
        else:
            circ.append(GatePos(gate=gate, target=target))
    circ = np.tile(circ, layer_count)
    circ = [array_to_gate_pos(gate) for gate in circ]
    return circ
