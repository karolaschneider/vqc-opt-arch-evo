"""Runner"""

""" 
- create an environment and agent(s) 
- run the game for a number of generations
- evolve the agents over time by selecting the best performers 
  and applying mutation and recombination.
"""

import copy
import random
from math import ceil, log2
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch import nn
from gym.spaces import Discrete
from pettingzoo.utils import AECEnv

from env import CoinGame
from vqc_layer import VQCLayer
from vqc_gate import VQCGate, create_random_circ, GatePos, create_SEL_circ


def create_env():
    """Creates the environment"""
    env = CoinGame(3, 3)
    return env

def initialize_random_agents(num_agents: int, env: CoinGame, circ_type: str, layer: int, num_gates: int) -> List[nn.Module]:
    """
    Returns the first randomly generated agents
    """
    agents = []

    # get size of observation space
    obsspace: Dict[str, np.ndarray[float, Any]] = env.observation_space("player_1")  # type: ignore
    obs = obsspace.get("observation")
    if obs is None:
        raise Exception("Fehler: Key Observation gibt es nicht im Observationspace")
    obs_num_elements = obs.shape[0] * obs.shape[1] * obs.shape[2]

    # size of actionspace: actionspace.n
    actionspace: Discrete = env.action_space("player_1")  # type: ignore

    # get number of qubits depending on size of action space and observation space
    # enough qubits so that the VQC can both represent the complexity of the action space and to process the observable elements
    num_qubits = max(actionspace.n, ceil(log2(obs_num_elements)))

    if circ_type == "layer":
        for _ in range(num_agents):
            # create the agent as VQC (layer-level)
            agent = VQCLayer(num_qubits, layer, actionspace.n, circ_type)
            agents.append(agent)
    elif circ_type == "prototype":
        for _ in range(num_agents):
            # create the agent as VQC (prototype-level)
            agent_list = create_random_circ(num_gates, num_qubits, layer)
            #agent_list = create_SEL_circ(num_qubits, layer) # for testing
            agent = VQCGate(num_qubits, agent_list, actionspace.n, circ_type, layer)
            agents.append(agent)
    elif circ_type == "gate":
        for _ in range(num_agents):
            # create the agent as VQC (gate-level)
            agent_list = create_random_circ(num_gates, num_qubits)
            #agent_list = create_SEL_circ(num_qubits, layer) # for testing
            agent = VQCGate(num_qubits, agent_list, actionspace.n, circ_type)
            agents.append(agent)

    return agents


def policy(observation_space: Dict[str, np.ndarray], agent: torch.nn.Module, gate_list: Union[List[Any], None] = None):
    """Policy function (decides the action)"""
    obs = observation_space.get("observation")
    if obs is None:
        raise Exception("Error: Key 'observation' not found in observation_space")
    
    action_mask = observation_space.get("action_mask")
    if action_mask is None:
        raise Exception("Error: Key 'action_mask' not found in observation_space")

    # Call agent with observation
    # agent returns values in range [0,1]
    obs_tensor = torch.tensor(np.array([obs.flatten()]), dtype=torch.float32)
    if agent.circ_type == "prototype" or agent.circ_type == "gate":
        action_values = agent(obs_tensor, gate_list)
    elif agent.circ_type == "layer":
        action_values = agent(obs_tensor)
    else:
        raise ValueError(f"Invalid circ_type: {agent.circ_type}")

    # Apply softmax to convert to probabilities
    action_values = torch.softmax(action_values, dim=1)

    # Apply action mask to ensure valid actions
    action_mask_tensor = torch.tensor(action_mask, dtype=action_values.dtype)
    act = action_values * action_mask_tensor
    action = torch.argmax(act)  # Pick action with highest value

    # print info if invalid action is chosen
    if action_mask[int(action.item())] == 0:
        print(f"action mask: {action_mask}")
        print(f"action_values: {action_values}")
        print(f"act: {act}")
        print(f"action: {action}")
        print(f"action.item() (the return of policy): {action.item()}")
    
    return action.item()


def run_multi_agent(
    env: CoinGame, agents: Dict[str, nn.Module], seed: int, render: bool = False
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    For every agent, run a whole episode n times
    """
    # reset environment and create empty scores list for the run
    scores = {"player_1": 0.0, "player_2": 0.0}
    coin_total = {"player_1": 0.0, "player_2": 0.0}
    other_coin_total = {"player_1": 0.0, "player_2": 0.0}
    own_coin_total = {"player_1": 0.0, "player_2": 0.0}
    env.reset(seed)
    for a in env.agent_iter():
        """ print("-------------------------------------")
        print(f"Agent {a}") """
        agent = agents.get(a)
        if agent is None:
            raise Exception(f"Es gibt keinen Agent für {a}")
        observation, reward, termination, truncation, _ = env.last()
        scores[a] = reward + scores.get(a, 0)
        if reward == -2:
            other_coin_total["player_1" if a == "player_2" else "player_2"] += 1
        elif reward == +1:
            coin_total[a] += 1

        if observation is None:
            raise Exception("Fehler: Observation darf nicht None sein")

        if termination or truncation:
            action = None
        else:
            # action taken using the policy function
            # differ between VQCLayer and VQCGate
            if agent.circ_type == "layer":
                action = policy(observation, agent)
            elif agent.circ_type in ["prototype", "gate"]:
                action = policy(observation, agent, agent.gate_list)
            else:
                raise ValueError(f"Invalid circ_type: {agent.circ_type}")

        # Render Env and take Action in Env
        env.step(action, render) # so that render can be set to true for the best agent in the last generation

        # calculate total own coins
        own_coin_total = {k: coin_total[k] - v for (k, v) in other_coin_total.items()}
    return scores, coin_total, own_coin_total


def create_new_gen(agents: List[nn.Module], scores: List[float], selection_type: str, tournament_size: int, num_top_indices: int, mutation_rate: float, param_mutation_power: float, arch_mutation_power: float, num_arch_mut: Optional[int] = None):
    """Creates a new generation of agents keeping the best agent and creating the rest through crossover and mutation"""
    children: List[nn.Module] = []
    #print("Scores of agents: " + ', '.join(f"({i}): {scores[i]}" for i in range(len(agents))))

    # Best agent survives
    if(agents[0].circ_type == "layer"):
        sizes = [-agents[i].num_layers for i in range(len(agents))]
    elif(agents[0].circ_type in ["prototype", "gate"]):
        sizes = [-agents[i].num_gates for i in range(len(agents))]
    else:
        raise ValueError("invalid circuit type")
    sorted_indices = np.lexsort((sizes, scores))
    best_agent = sorted_indices[-1]
    #print("Index of best agent:", best_agent)
    children.append(agents[best_agent])

    # create new gen by crossover and mutation
    while len(children) < len(agents):
        # Select parents
        if selection_type == "tournament":
            index_parent_m = tournament_selection(agents, scores, tournament_size)
            index_parent_f = tournament_selection(agents, scores, tournament_size)
            # make sure to choose different parents
            while(index_parent_m == index_parent_f):
                index_parent_f = tournament_selection(agents, scores, tournament_size)
        elif selection_type == "truncation":
            index_parent_m = truncation_selection(agents, scores, num_top_indices)
            index_parent_f = truncation_selection(agents, scores, num_top_indices)
            # make sure to choose different parents
            while(index_parent_m == index_parent_f):
                index_parent_f = truncation_selection(agents, scores, num_top_indices)
        else:
            raise ValueError(f"Invalid selection_type: {selection_type}")
        
        parent_m = agents[index_parent_m]
        parent_f = agents[index_parent_f]

        # Evolutionary Algorithms
        if agents[0].circ_type == "layer":
            # Crossover recombination
            child_1, child_2 = crossover_layer_level(parent_m, parent_f)
            # Mutation
            child_1 = mutate_layer_level(child_1, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
            child_2 = mutate_layer_level(child_2, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
        elif agents[0].circ_type in ["prototype", "gate"]:
            # Crossover recombination
            child_1, child_2 = crossover_gate_level(parent_m, parent_f)
            # Mutation
            child_1 = mutate_gate_level(child_1, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
            child_2 = mutate_gate_level(child_2, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
        else:
            raise ValueError(f"Invalid circ_type: {agents[0].circ_type}")
        #print(f">> Parents {index_parent_m} & {index_parent_f} created children and they were possibly mutated")

        children.append(child_1)
        children.append(child_2)
    # remove last child if there are more children than agents in the previous generation
    if len(children) > len(agents):
        children.pop()
    
    return children


def create_new_gen_mut_only(agents: List[nn.Module], scores: List[float], selection_type: str, tournament_size: int, num_top_indices: int, param_mutation_power: float, arch_mutation_power: float, num_arch_mut: Optional[int] = None):
    """Creates a new generation of agents keeping the best agent and creating the rest through mutation"""
    children: List[nn.Module] = []
    #print("Scores of agents: " + ', '.join(f"({i}): {scores[i]}" for i in range(len(agents))))

    # Best agent survives
    if(agents[0].circ_type == "layer"):
        sizes = [-agents[i].num_layers for i in range(len(agents))]
    elif(agents[0].circ_type in ["prototype", "gate"]):
        sizes = [-agents[i].num_gates for i in range(len(agents))]
    else:
        raise ValueError("invalid circuit type")
    sorted_indices = np.lexsort((sizes, scores))
    best_agent = sorted_indices[-1]
    #print("Index of best agent:", best_agent)
    children.append(agents[best_agent])

    # no mutation rate for mutation only
    mutation_rate = 1

    # create new gen by mutation
    while len(children) < len(agents):
        if selection_type == "tournament":
            selected_index = tournament_selection(agents, scores, tournament_size)
        elif selection_type == "truncation":
            selected_index = truncation_selection(agents, scores, num_top_indices)
        else:
            raise ValueError(f"Invalid selection_type: {selection_type}")
        #print(f"Mutate agent with index: {selected_index}")
        child_agent = copy.deepcopy(agents[selected_index])
        if agents[selected_index].circ_type == "layer":
            child_agent = mutate_layer_level(child_agent, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
        elif agents[selected_index].circ_type in ["prototype", "gate"]:
            child_agent = mutate_gate_level(child_agent, mutation_rate, param_mutation_power, arch_mutation_power, num_arch_mut)
        else:
            raise ValueError(f"Invalid circ_type: {agents[0].circ_type}")
        children.append(child_agent)
    
    return children


def crossover_layer_level(agent_m: nn.Module, agent_f: nn.Module):
    """ Creates new agents for future generations through single-point crossover for layer-based VQC """
    #print("Crossover on layer-level")
    if min(agent_m.num_layers, agent_f.num_layers) <= 2:
        cut_point = 1
    else:
        cut_point = random.randint(1, min(agent_m.num_layers, agent_f.num_layers)-1)
    size_child_1 = agent_f.num_layers
    size_child_2 = agent_m.num_layers
    child_1 = VQCLayer(agent_m.num_qubits, size_child_1, agent_m.action_space, agent_m.circ_type)
    child_2 = VQCLayer(agent_m.num_qubits, size_child_2, agent_m.action_space, agent_m.circ_type)

    weights_m = None
    bias_m = None
    weights_f = None
    bias_f = None

    # Extract weights and biases from parent_m
    for name, param in agent_m.named_parameters():
        if name == "weights":
            weights_m = param.data
        if name == "bias":
            bias_m = param.data
    # Extract weights and biases from parent_f
    for name, param in agent_f.named_parameters():
        if name == "weights":
            weights_f = param.data
        if name == "bias":
            bias_f = param.data

    # Ensure weights and biases are found
    if weights_m is None or bias_m is None or weights_f is None or bias_f is None:
        raise ValueError("Weights or biases not found in the agents")    

    # Recombine weights at cutting point
    # child_1 gets bias of parent_m and child_2 gets bias of parent_f
    for name, param in child_1.named_parameters():
        if name == "weights":
            param.data = torch.cat([weights_m[:cut_point], weights_f[cut_point:]])
        if name == "bias":
            param.data = bias_m
    for name, param in child_2.named_parameters():
        if name == "weights":
            param.data = torch.cat([weights_f[:cut_point], weights_m[cut_point:]])
        if name == "bias":
            param.data = bias_f
    return child_1, child_2


def arch_mut_layer_level(agent: nn.Module):
    """ Mutates the architecture of the agent for layer-based VQC """
    ADD_LAYER = 0
    REMOVE_LAYER = 1
    mode = ADD_LAYER # default: add layer
    if(agent.num_layers > 1):
        # if more than one layer -> 50/50 if add or remove
        if(np.random.rand() < 0.5):
            mode = REMOVE_LAYER
    if(mode == ADD_LAYER):
        position_to_add = random.randint(0, agent.num_layers) 
        agent.num_layers += 1
        #print("(+) Mutation: Added layer at position", position_to_add)
        # add new layer of weights
        for name, param in agent.named_parameters():
            if name == "weights":
                original_weights = param.data
                #print(f"original weights: {original_weights}")
                layer_to_add = torch.rand(size=(1, agent.num_qubits, 3)) * 2 * torch.pi - torch.pi
                #print(f"layer to add: {layer_to_add}")
                param.data = torch.cat([original_weights[:position_to_add], layer_to_add, original_weights[position_to_add:]])
                #print(f"new weights: {param.data}")
    elif(mode == REMOVE_LAYER):
        layer_to_remove = random.randint(0, agent.num_layers-1)
        agent.num_layers -= 1 
        #print(f"Layer to remove is in range: {0} - {agent.num_layers}")
        #print("(-) Mutation: Removed layer at position", layer_to_remove)
        # get the original agent's weights
        for name, param in agent.named_parameters():
            if name == "weights":
                original_weights = param.data
                #print(f"original weights: {original_weights}")
                param.data = torch.cat([original_weights[:layer_to_remove], original_weights[layer_to_remove + 1:]])
                #print(f"new weights: {param.data}")
    return agent

def custom_round(value):
    """ Custom rounding function """
    if value < 0:
        return int(value - 0.5)
    return int(value + 0.5)

# mutates the agent in place; agent is returned for consistency purposes
def mutate_layer_level(agent: nn.Module, mutation_rate: float, param_mutation_power: float, arch_mutation_power: float , num_arch_mut: Optional[int] = None):
    """ Creates new agents for future generations through mutation for layer-based VQC """
    #print("Mutation on layer-level")
    # decide if architecture mutation happens
    if(np.random.rand() < mutation_rate):
        #print("Architecture mutation happens")
        # different types to pass architecture mutation power tested
        if num_arch_mut is not None:
            num_mutations = num_arch_mut
        else:
            num_mutations = max(1, custom_round(agent.num_layers * arch_mutation_power))
            #print(f"mut_power: {arch_mutation_power} | num_layers: {agent.num_layers} -> Number of mutations: {num_mutations}")
        for _ in range (num_mutations):
            agent = arch_mut_layer_level(agent)

    # decide if parameter mutation happens
    if(np.random.rand() < mutation_rate):
        #print("Parameter mutation happens")
        agent = parameter_mutation(agent, param_mutation_power)
    
    return agent  

def normalize_angle(angle_tensor: torch.Tensor):
    """ Normalize the angle tensor to [-π, π] """
    # Normalize angle to [0, 2π]
    angle_tensor = angle_tensor % (2 * torch.pi)
    # If angle is greater than π, map it to [-π, π]
    angle_tensor[angle_tensor >= torch.pi] -= 2 * torch.pi
    return angle_tensor


# mutates the agent in place; agent is returned for consistency purposes
def parameter_mutation(agent: nn.Module, param_mutation_power: float): 
    """ Mutates the parameters of the agent """
    for name, param in agent.named_parameters():
        noise = torch.randn_like(param) * param_mutation_power
        param.data += noise
        if name == "weights":
            #print(f"before remapping: {param.data}")
            param.data = normalize_angle(param.data)
            #print(f"after remapping: {param.data}")
    return agent


# only works for circuits with same number of layers!
def crossover_gate_level(agent_m: nn.Module, agent_f: nn.Module):
    """ Creates new agents for future generations through single-point crossover for gate-based or prototype-based VQC """
    #print(f"Crossover on {agent_m.circ_type}-level")

    num_gates_agent_m = -1
    num_gates_agent_f = -1
    prototype_m = None
    prototype_f = None
    child_1_list = None
    child_2_list = None
    weights_m = None
    weights_f = None
    bias_m = None
    bias_f = None

    if agent_m.circ_type == "prototype":
        num_gates_agent_m = int(len(agent_m.gate_list)/agent_m.num_layers)
        num_gates_agent_f = int(len(agent_f.gate_list)/agent_f.num_layers)
        prototype_m = agent_m.gate_list[:num_gates_agent_m]
        prototype_f = agent_f.gate_list[:num_gates_agent_f]
        #print(f"prototype_m: {prototype_m}")
        #print(f"prototype_f: {prototype_f}")
    
    elif agent_m.circ_type == "gate":
        num_gates_agent_m = len(agent_m.gate_list)
        num_gates_agent_f = len(agent_f.gate_list)
    
    if (num_gates_agent_m == -1) or (num_gates_agent_f == -1):
        raise ValueError("Number of gates not found in the agents")
    
    #print(f"cutting point is in range {1} and {min(num_gates_agent_m, num_gates_agent_f) - 1}")
    if min(num_gates_agent_m, num_gates_agent_f) <= 2:
        cut_point = 1
    else:
        cut_point = random.randint(1, min(num_gates_agent_m, num_gates_agent_f) - 1)
    #print(f"Cutting point: {cut_point}")
    if agent_m.circ_type == "prototype":
        if (prototype_m is None or prototype_f is None):
            raise ValueError("Prototype not found in the agents")
        child_1_list = np.concatenate((prototype_m[:cut_point], prototype_f[cut_point:]))
        child_2_list = np.concatenate((prototype_f[:cut_point], prototype_m[cut_point:]))
        #print(f"child_1_list: {child_1_list}")
        #print(f"child_2_list: {child_2_list}")
        child_1_list = np.tile(child_1_list, agent_m.num_layers)
        child_2_list = np.tile(child_2_list, agent_f.num_layers)
        # Convert the NumPy arrays to lists of GatePos objects
        child_1_list = [GatePos(pos.gate, pos.target, pos.control) for pos in child_1_list]
        child_2_list = [GatePos(pos.gate, pos.target, pos.control) for pos in child_2_list]
        #print(f"child_1_list: {child_1_list}")
        #print(f"child_2_list: {child_2_list}")
    elif agent_m.circ_type == "gate":
        child_1_list = np.concatenate((agent_m.gate_list[:cut_point], agent_f.gate_list[cut_point:]))
        child_2_list = np.concatenate((agent_f.gate_list[:cut_point], agent_m.gate_list[cut_point:]))
        # Convert the NumPy arrays to lists of GatePos objects
        child_1_list = [GatePos(pos.gate, pos.target, pos.control) for pos in child_1_list]
        child_2_list = [GatePos(pos.gate, pos.target, pos.control) for pos in child_2_list]

    for name, param in agent_m.named_parameters():
        if name == "weights":
            weights_m = param.data
        if name == "bias":
            bias_m = param.data
    for name, param in agent_f.named_parameters():
        if name == "weights":
            weights_f = param.data
        if name == "bias":
            bias_f = param.data

    if (weights_m is None or weights_f is None):
        raise ValueError("Weights not found in the agents")
    if (bias_m is None or bias_f is None):
        raise ValueError("Biases not found in the agents")

    if agent_m.circ_type == "prototype":
        if (child_1_list is None or child_2_list is None):
            raise ValueError("Child list not correctly created")
        child_1 = VQCGate(agent_m.num_qubits, child_1_list, agent_m.action_space, agent_m.circ_type, agent_m.num_layers)
        child_2 = VQCGate(agent_m.num_qubits, child_2_list, agent_m.action_space, agent_m.circ_type, agent_m.num_layers)
        weights_m = weights_m.numpy().flatten()
        weights_f = weights_f.numpy().flatten()
        # Split the weights into as many pieces as there are layers
        weights_m_split = np.split(weights_m, agent_m.num_layers)
        weights_f_split = np.split(weights_f, agent_f.num_layers)
        child_1_weights_split = []
        child_2_weights_split = []

        # Perform crossover on each pair of corresponding pieces
        for weights_m_piece, weights_f_piece in zip(weights_m_split, weights_f_split):
            child_1_weights_piece = np.concatenate((weights_m_piece[:cut_point], weights_f_piece[cut_point:]))
            child_2_weights_piece = np.concatenate((weights_f_piece[:cut_point], weights_m_piece[cut_point:]))
            child_1_weights_split.append(child_1_weights_piece)
            child_2_weights_split.append(child_2_weights_piece)

        # Concatenate the pieces back together
        child_1_weights = np.concatenate(child_1_weights_split)
        child_2_weights = np.concatenate(child_2_weights_split)

        for name, param in child_1.named_parameters():
            if name == "weights":
                param.data = torch.from_numpy(child_1_weights)
            if name == "bias":
                param.data = bias_m
        for name, param in child_2.named_parameters():
            if name == "weights":
                param.data = torch.from_numpy(child_2_weights)
            if name == "bias":
                param.data = bias_f
    elif agent_m.circ_type == "gate":
        if (child_1_list is None or child_2_list is None):
            raise ValueError("Child list not correctly created")
        child_1 = VQCGate(agent_m.num_qubits, child_1_list, agent_m.action_space, agent_m.circ_type)
        child_2 = VQCGate(agent_m.num_qubits, child_2_list, agent_m.action_space, agent_m.circ_type)
        for name, param in child_1.named_parameters():
            if name == "weights":
                param.data = torch.cat([weights_m[:cut_point], weights_f[cut_point:]])
            if name == "bias":
                param.data = bias_m
        for name, param in child_2.named_parameters():
            if name == "weights":
                param.data = torch.cat([weights_f[:cut_point], weights_m[cut_point:]])
            if name == "bias":
                param.data = bias_f
    else:
        raise ValueError(f"Invalid circ_type: {agent_m.circ_type}")
    return child_1, child_2


def arch_mut_gate_level(agent: nn.Module):
    """ Mutates the architecture of the agent for gate-based or prototype-based VQC """
    num_gates = -1
    if agent.circ_type == "prototype":
        num_gates = agent.num_gates/agent.num_layers
    elif agent.circ_type == "gate":
        num_gates = agent.num_gates
    if num_gates == -1:
        raise ValueError("Number of gates not found in the agents")
    if num_gates > 1:
        modes = {
            'add_gate': add_gate,
            'remove_gate': remove_gate,
            'swap_gate': swap_gate
        }
    else:
        modes = {
            'add_gate': add_gate,
            'swap_gate': swap_gate
        }
    mode_name = np.random.choice(list(modes.keys()))
    #print(f"Mutation mode: {mode_name}")
    # apply the chosen mutation
    agent = modes[mode_name](agent)
    return agent


# mutates the agent in place; agent is returned for consistency purposes
def mutate_gate_level(agent: nn.Module, mutation_rate: float, param_mutation_power: float, arch_mutation_power: float, num_arch_mut: Optional[int] = None):
    """ Creates new agents for future generations through mutation for gate-based or prototype-based VQC """
    #print(f"Mutation on {agent.circ_type}-level")
    # decide if architecture mutation happens
    if(np.random.rand() < mutation_rate):
        #print("Architecture mutation happens")
        num_mutations = -1
        if num_arch_mut is not None:
            num_mutations = num_arch_mut
        else:
            if agent.circ_type == "prototype":
                num_mutations = max(1, custom_round(agent.num_gates/agent.num_layers * arch_mutation_power))
                #print(f"mut_power: {arch_mutation_power} | num_gates per layer: {agent.num_gates/agent.num_layers} -> Number of mutations: {num_mutations}")
            elif agent.circ_type == "gate":
                num_mutations = max(1, custom_round(agent.num_gates * arch_mutation_power))
                #print(f"mut_power: {arch_mutation_power} | num_gates: {agent.num_gates} -> Number of mutations: {num_mutations}")
        if num_mutations == -1:
            raise ValueError("Number of mutations not found in the agents")
        for _ in range (num_mutations):
            agent = arch_mut_gate_level(agent)
        
    # decide if parameter mutation happens (weights and bias)
    if(np.random.rand() < mutation_rate):
        #print("Parameter mutation happens")
        agent = parameter_mutation(agent, param_mutation_power)
    return agent

def gate_pos_to_array(gate_pos: GatePos):
    """ Converts a GatePos object to a structured array """
    return (gate_pos.gate, gate_pos.target, gate_pos.control)

def array_to_gate_pos(array):
    """ Converts a structured array to a GatePos object """
    return GatePos(gate=array['gate'], target=array['target'], control=array['control'])

def add_gate(agent: nn.Module, pos: Optional[int] = None, target_qubit: Optional[int] = None):
    """ Adds a gate to the agent """
    gate_to_add = create_random_circ(1, agent.num_qubits, target_qubit = target_qubit)
    repetitions = agent.num_layers if agent.circ_type == "prototype" else 1
    index_to_add = int(pos if pos is not None else random.randint(0, agent.num_gates//repetitions))
    """ if(pos is None): # only print if its not called from switch_gate
        print(f"(+) Adding gate {gate_to_add} at position {index_to_add}")
        print(f"gate list before adding: {agent.gate_list}")
        print(f"parameter list before adding: {agent.weights}") """
    new_weights = []
    size_prototype = agent.num_gates/repetitions
    for x in range(repetitions):
        new_weight = (torch.rand(1) * 2 * torch.pi) - torch.pi
        new_weights.append(new_weight)
    concatenated_weights = agent.weights.clone()

    # Convert gate_list to a structured array
    dtype = [('gate', 'U10'), ('target', 'i4'), ('control', 'O')]
    gate_list_array = np.array([gate_pos_to_array(gate) for gate in agent.gate_list], dtype=dtype)
    gate_to_add_array = np.array([gate_pos_to_array(gate_to_add[0])], dtype=dtype)

    for i, weight in enumerate(new_weights):
        gate_list_array = np.insert(gate_list_array, int(index_to_add + size_prototype * i + i), gate_to_add_array, axis=0)
        concatenated_weights = torch.cat((concatenated_weights[:int(index_to_add + size_prototype * i + i)], new_weights[i], concatenated_weights[int(index_to_add+size_prototype * i + i):]), 0)
    setattr(agent, 'gate_list', [array_to_gate_pos(gate) for gate in gate_list_array])
    agent.weights = nn.Parameter(concatenated_weights, requires_grad=False)
    setattr(agent, 'num_gates', len(agent.gate_list))
    """ if(pos is None):
        print(f"gate list after adding: {agent.gate_list}")
        print(f"parameter list after adding: {agent.weights}") """
    return agent

def remove_gate(agent: nn.Module, pos: Optional[int] = None):
    """ Removes a gate from the agent """
    repetitions = agent.num_layers if agent.circ_type == "prototype" else 1
    size_prototype = agent.num_gates/repetitions
    index_to_remove = int(pos if pos is not None else random.randint(0, agent.num_gates//repetitions-1))
    """ if(pos is None): # only print if its not called from switch_gate
        print(f"(-) Removing gate at index {index_to_remove}")
        print(f"gate list before removing: {agent.gate_list}")
        print(f"parameter list before removing: {agent.weights}") """
    concatenated_weights = agent.weights.clone()

    dtype = [('gate', 'U10'), ('target', 'i4'), ('control', 'O')]
    gate_list_array = np.array([gate_pos_to_array(gate) for gate in agent.gate_list], dtype=dtype)
    for i in range(repetitions):
        gate_list_array = np.delete(gate_list_array, int(index_to_remove + size_prototype * i - i), axis=0)
        concatenated_weights = torch.cat((concatenated_weights[:int(index_to_remove + size_prototype * i - i)], concatenated_weights[int(index_to_remove + size_prototype * i - i) + 1:]))
    setattr(agent, 'gate_list', [array_to_gate_pos(gate) for gate in gate_list_array])
    agent.weights = nn.Parameter(concatenated_weights, requires_grad=False)
    setattr(agent, 'num_gates', len(agent.gate_list))
    """ if(pos is None):
        print(f"gate list after removing: {agent.gate_list}")
        print(f"parameter list after removing: {agent.weights}") """
    return agent

def swap_gate(agent: nn.Module):
    """ Swaps a gate in the agent """
    repetitions = agent.num_layers if agent.circ_type == "prototype" else 1
    index_to_swap = random.randint(0, len(agent.gate_list)//repetitions - 1)
    target_qubit = agent.gate_list[index_to_swap].target
    #print(f"(~) Swapping gate at index {index_to_swap}")
    #print(f"gate list before swapping: {agent.gate_list}")
    remove_gate(agent, index_to_swap)
    add_gate(agent, index_to_swap, target_qubit)
    #print(f"gate list after swapping: {agent.gate_list}")
    return agent


def tournament_selection(agents: List[nn.Module], scores: List[float], size: int):
    """ Selects the best agent from a random subset of agents """
    tournament = random.sample(range(len(agents)), size)
    #print("---Tournament Selection---")
    """ print("Score of agents in tournament: " + ', '.join(f"({i}): {scores[i]}" for i in tournament))
    if(agents[0].circ_type == "layer"):
        print("Size of agents in tournament: " + ', '.join(f"({i}): {agents[i].num_layers}" for i in tournament))
    elif(agents[0].circ_type in ["prototype", "gate"]):
        print("Size of agents in tournament: " + ', '.join(f"({i}): {agents[i].num_gates}" for i in tournament)) """
    if(agents[0].circ_type == "layer"):
        sizes = [-agents[i].num_layers for i in range(len(agents))]
    elif(agents[0].circ_type in ["prototype", "gate"]):
        sizes = [-agents[i].num_gates for i in range(len(agents))]
    else:
        raise ValueError("invalid circuit type")
    sorted_indices = np.lexsort((sizes, scores))
    tournament_sorted_indices = [i for i in sorted_indices if i in tournament]
    #print(f"Index of selected agent: {tournament_sorted_indices[-1]}")
    #print("--------------------------")
    return tournament_sorted_indices[-1]

def truncation_selection(agents: List[nn.Module], scores: List[float], size: int):
    """ Selects an agent randomly from the subset of top-performing agents """
    if(agents[0].circ_type == "layer"):
        sizes = [-agents[i].num_layers for i in range(len(agents))]
    elif(agents[0].circ_type in ["prototype", "gate"]):
        sizes = [-agents[i].num_gates for i in range(len(agents))]
    else:
        raise ValueError("invalid circuit type")
    sorted_indices = np.lexsort((sizes, scores))
    top_indices = np.sort(sorted_indices[-size:])
    chosen_index = random.choice(top_indices.tolist())
    #print("---Truncation Selection---")
    """ print("Score of agents: " + ', '.join(f"({i}): {scores[i]}" for i in top_indices))
    if(agents[0].circ_type == "layer"):
        print("Size of agents: " + ', '.join(f"({i}): {agents[i].num_layers}" for i in top_indices))
    elif(agents[0].circ_type in ["prototype", "gate"]):
        print("Size of agents: " + ', '.join(f"({i}): {agents[i].num_gates}" for i in top_indices))
    print("Index of selected agent: ", chosen_index) """
    #print("--------------------------")
    return chosen_index
