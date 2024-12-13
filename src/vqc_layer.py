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


# QC helper functions (like rotations, entanglement, etc.)
def rotate(w_layer: Tensor):
    """Rotations"""
    for i, w_wire in enumerate(w_layer):
        theta_x, theta_y, theta_z = w_wire
        qml.Rot(theta_x, theta_y, theta_z, wires=i)

def entangle(n_qubits: int):
    """Entangles the qubits"""
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    qml.CNOT(wires=[n_qubits - 1, 0])


# VQC circuit Layer-Based
class VQCLayer(nn.Module):
    """Variational quantum circuit"""

    # action space depends on env
    def __init__(self, num_qubits: int, num_layers: int, action_space: int, circ_type: str):
        super().__init__()

        # layers and qubits
        self.num_layers = num_layers
        self.num_qubits = num_qubits

        # action space dimension
        self.action_space = action_space

        self.circ_type = circ_type

        # create qnode with choosen device
        self.device = qml.device("default.qubit", wires=range(num_qubits))
        self.qnode = qml.QNode(self.circuit, self.device, interface="torch")

        # parameters randomly initialized
        self.weights = nn.Parameter(
            (torch.rand(size=(self.num_layers, self.num_qubits, 3)) * 2 * torch.pi) - torch.pi,
            requires_grad=False,
        )
        # bias randomly initialized
        self.bias = nn.Parameter((torch.rand(self.action_space) * 0.01), requires_grad=False)

    def circuit(self, weights: Tensor, input_data: Tensor):
        """Builds the circuit"""
        qml.AmplitudeEmbedding(
            features=input_data, wires=range(self.num_qubits), pad_with=0.3, normalize=True
        )
        for i in range(self.num_layers):
            entangle(self.num_qubits)
            rotate(weights[i])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.action_space)]

    def forward(self, input_data: Tensor) -> Tensor:
        """Forward pass"""
        results = []
        for x_i in input_data:
            result = self.qnode(self.weights, x_i)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result)
            results.append(result)
        stacked_results = torch.stack(results)
        biased_results = stacked_results + self.bias
        return biased_results

    def show(self, input_data: Tensor):
        """Prints the circuit"""
        draw = qml.draw(self.qnode)
        print(draw(self.weights, input_data))
