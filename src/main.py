"""Main python file"""
import argparse
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import runner
from vqc_gate import count_gates

GENERATIONS = 200
NUM_AGENTS = 250 # min 3 wg. NUM_TOP_AGENTS

# playceholder values; these are specified in the parse_args function depending on the circuit type
NUM_GATES = 0
NUM_LAYERS = 0

""" Selection """
SELECTION_TYPE = "tournament" # truncation or tournament
TOURNAMENT_SIZE = max(int(NUM_AGENTS * 0.4), 2) # 40 percent of population, but at least 2
NUM_TOP_AGENTS = 5 # number of top agents to be selected for reproduction


""" Mutation """
MUTATION_RATE = 0.1

## MUTATION POWER
# default: use static mutation power
PARAM_MUTATION_POWER = 0.01 # mutation power for parameters

ARCH_MUTATION_POWER = 0.01 # relict, is not used anymore: mutation power for architecture
# set fixed number of architecture mutations with -a <num_arch_mut>; disables the use of ARCH_MUTATION_POWER
NUM_ARCH_MUT = None # number of architecture mutations

# set TRUE for more exploitation in last quarter of generations; default: False
EXPLOIT = False


""" Dynamic Hyperparameters """#
# per default not dynamic
# use -d to enable dynamic selection and mutation power
USE_DYNAMIC = False 

TOURN_START = max(int(NUM_AGENTS * 0.05), 2)
TOURN_END = max(int(NUM_AGENTS * 0.4), 2)

START_PARAM_MUT_POW = 0.05
END_PARAM_MUT_POW = 0.05

START_ARCH_MUT_POW = 0.01
END_ARCH_MUT_POW = 0.01


""" Parse args """
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("-s", "--seed", type=int, help="Seed", required=True)
    parser.add_argument("-t", "--type", type=str, choices=['layer', 'prototype', 'gate'], help="Type of circuit architecture", required=True)
    parser.add_argument("-e", "--evolution", type=str, choices=['mut-recomb', 'mut-only'], help="Evolutionary strategy", required=True)
    parser.add_argument("-a", "--num-arch-mut", type=int, help="Number of architecture mutations", required=False)
    parser.add_argument("-d", "--dynamic-mut-pow", action="store_true", help="Enable dynamic mutation power")
    parser.add_argument("-x", "--exploit", action="store_true", help="Enable high exploitation in last quarter")
    args = parser.parse_args()

    global USE_DYNAMIC
    USE_DYNAMIC = args.dynamic_mut_pow

    global EXPLOIT
    EXPLOIT = args.exploit

    global NUM_ARCH_MUT
    if args.num_arch_mut is not None:
        NUM_ARCH_MUT = args.num_arch_mut

    # has to be set according to the circuit type
    # NUM_GATES: number of gates a gate-level and prototype-level circuit (per layer) should start with
    # NUM_LAYERS: number of layers a layer-level and prototype-level circuit should start with
    global NUM_GATES, NUM_LAYERS
    if args.type == 'layer':
        NUM_GATES = 0
        NUM_LAYERS = 1
    elif args.type == 'gate':
        NUM_GATES = 70
        NUM_LAYERS = 0
    elif args.type == 'prototype':
        NUM_GATES = 18
        NUM_LAYERS = 8
    
    return args.seed, args.type, args.evolution


""" Functions used for dynamic hyperparameters """
# function for dynamic mutation power
""" def calculate_param_mutation_power(start: float, end: float, current_generation: int):
    if(GENERATIONS <= 2 or current_generation + 1 == GENERATIONS):
        return end
    # Calculate the relative position of the current generation
    relative_position = (current_generation) / (GENERATIONS - 2)

    # Linearly interpolate the mutation power for the current generation
    mutation_power = start + (end - start) * relative_position

    return mutation_power """

def interpolate_generations(start: float, end: float, current_generation: int) -> float:
    """Calculates the value for the current generation based on a start and end value"""
    if GENERATIONS <= 2 or current_generation + 1 == GENERATIONS:
        return end
    step_size = (end - start) / (GENERATIONS - 2)
    current_value = start + step_size * current_generation
    return current_value



def main(seed: int, circ_type: str, evolution_type: str):
    """Main entrypoint of the program"""
    global NUM_TOP_AGENTS
    global SELECTION_TYPE
    global TOURNAMENT_SIZE
    global MUTATION_RATE

    """ Create the environment """
    env = runner.create_env()

    """ Seed everything """
    def seed_everything(seed_value: int):
        env.reset(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.use_deterministic_algorithms(True)

        return np.random.default_rng(seed_value)

    seed_everything(seed_value=seed)

    """ Initialize agent(s) """
    agents = runner.initialize_random_agents(num_agents=NUM_AGENTS, env=env, circ_type=circ_type, layer=NUM_LAYERS, num_gates=NUM_GATES)

    # set mut_rate to None for mut-only
    if evolution_type == "mut-only":
        MUTATION_RATE = "-"

    """ Params for logging """
    # final parameters
    averages = []
    averages_top = []

    # parameters for dataframe
    current_generation_d = []
    generations_d = []
    test_type = []
    seed_d = []
    selection_type = []
    evo_type = []
    population_size_d = []
    mutation_rate_d = []
    param_mutation_power_d = []
    num_arch_mut_d = [] 
    arch_mutation_power_d = []
    top_limit_d = []
    avg_score_d = []
    std_dev_score_d = []
    coin_avg_d = []
    std_dev_coin_total_d = []
    own_coin_avg_d = []
    std_dev_own_coin_total_d = []
    
    layer_avg_d = []
    best_agent_layer_count_d = []
    second_best_agent_layer_count_d = []
    third_best_agent_layer_count_d = []

    gate_avg_per_layer_d = []
    layer_num_d = []
    best_agent_gate_count_per_layer_d = []
    second_best_agent_gate_count_per_layer_d = []
    third_best_agent_gate_count_per_layer_d = []

    gate_avg_d = []
    best_agent_gate_count_d = []
    second_best_agent_gate_count_d = []
    third_best_agent_gate_count_d = []

    time_d = []
    best_agent_score_d = []
    best_agent_coin_total_d = []
    best_agent_own_coin_total_d = []
    second_best_agent_score_d = []
    second_best_agent_coin_total_d = []
    second_best_agent_own_coin_total_d = []
    third_best_agent_score_d = []
    third_best_agent_coin_total_d = []
    third_best_agent_own_coin_total_d = []
    best_agent_parameterized_gates_d = []

    dt_string = None
    df = pd.DataFrame()

    # Evolutionary Loop
        # in each gen  :
            # agent evaluation
            # selection & reproduction
            # logging and output
        # after each gen: collect various metrics and store them in a pandas DataFrame 
        # (DataFrame is then written to a CSV)
    for generation in range(GENERATIONS):
        start_time = time.time()
        #print(f"\n<<< Generation {generation + 1} of {GENERATIONS} >>>")

        # Reset scores and coin totals
        scores = [] * NUM_AGENTS
        coin_total = [] * NUM_AGENTS
        own_coin_total = [] * NUM_AGENTS

        for i, agent in enumerate(agents):
            #print(f"-- agent {i+1} of {NUM_AGENTS} --")
            # agent.show(torch.rand((6)))

            # run the agent against itself and get metrics
            agent_1 = agents[i]
            agent_2 = agents[i]
            score_dict, coin_total_dict, own_coin_total_dict = runner.run_multi_agent(
                env, {"player_1": agent_1, "player_2": agent_2}, seed=seed
            )

            # append all data collected from run
            scores.append(score_dict["player_1"] + score_dict["player_2"])
            coin_total.append(coin_total_dict["player_1"] + coin_total_dict["player_2"])
            own_coin_total.append(
                own_coin_total_dict["player_1"] + own_coin_total_dict["player_2"]
            )
        
        # calculate Averages
        scores = np.array(scores)
        score_avg = np.mean(scores)
        coin_total = np.array(coin_total)
        coin_avg = np.mean(coin_total)
        own_coin_total = np.array(own_coin_total)
        own_coin_avg = np.mean(own_coin_total)

        # sort by score and size
        if(agents[0].circ_type == "layer"):
            sizes = [-agents[i].num_layers for i in range(len(agents))]
        elif(agents[0].circ_type in ["prototype", "gate"]):
            sizes = [-agents[i].num_gates for i in range(len(agents))]
        else:
            raise ValueError("invalid circuit type")
        sorted_indices = np.lexsort((sizes, scores))
        #print(f"scores: {scores}")
        #print(f"sizes: {sizes}")
        #print(f"sorted_indices: {sorted_indices}")

        # take the indices of the top n agents
        top_indices = sorted_indices[-int(NUM_TOP_AGENTS) :]

        # best scores (agents sorted by score and size)
        best_agent_score_d.append(scores[sorted_indices[-1]])
        second_best_agent_score_d.append(scores[sorted_indices[-2]])
        third_best_agent_score_d.append(scores[sorted_indices[-3]])

        # best coin totals
        best_agent_coin_total_d.append(coin_total[sorted_indices[-1]])
        second_best_agent_coin_total_d.append(coin_total[sorted_indices[-2]])
        third_best_agent_coin_total_d.append(coin_total[sorted_indices[-3]])
    
        # best own coin totals
        best_agent_own_coin_total_d.append(own_coin_total[sorted_indices[-1]])
        second_best_agent_own_coin_total_d.append(own_coin_total[sorted_indices[-2]])
        third_best_agent_own_coin_total_d.append(own_coin_total[sorted_indices[-3]])

        # best agent parameterized gates
        if circ_type == "layer":
            best_agent_parameterized_gates_d.append(agents[sorted_indices[-1]].num_qubits * 3 * agents[sorted_indices[-1]].num_layers)
        elif circ_type in ["prototype", "gate"]:
            best_agent_parameterized_gates_d.append(count_gates(agents[sorted_indices[-1]].gate_list, ['RX', 'RY', 'RZ']))
        else:
            best_agent_parameterized_gates_d.append("-")

        # data that is only needed with layer concept
        if circ_type == "layer":
            layer_avg = sum([agent.num_layers for agent in agents], start=0) / len(agents)
            layer_avg_d.append(layer_avg)
            best_agent_layer_count_d.append(agents[sorted_indices[-1]].num_layers)
            second_best_agent_layer_count_d.append(agents[sorted_indices[-2]].num_layers)
            third_best_agent_layer_count_d.append(agents[sorted_indices[-3]].num_layers)

        # data that is only needed with prototype concept
        if circ_type == "prototype":
            gate_avg_per_layer = sum([agent.num_gates/agent.num_layers for agent in agents], start=0) / len(agents)
            layer_num = sum([agent.num_layers for agent in agents], start=0) / len(agents)
            gate_avg_per_layer_d.append(gate_avg_per_layer)
            layer_num_d.append(layer_num)
            best_agent_gate_count_per_layer_d.append(agents[sorted_indices[-1]].num_gates/agents[sorted_indices[-1]].num_layers)
            second_best_agent_gate_count_per_layer_d.append(agents[sorted_indices[-2]].num_gates/agents[sorted_indices[-2]].num_layers)
            third_best_agent_gate_count_per_layer_d.append(agents[sorted_indices[-3]].num_gates/agents[sorted_indices[-3]].num_layers)

        # data that is only needed with gate concept 
        if circ_type == "gate": 
            gate_avg = sum([agent.num_gates for agent in agents], start=0) / len(agents)
            gate_avg_d.append(gate_avg)
            best_agent_gate_count_d.append(agents[sorted_indices[-1]].num_gates)
            second_best_agent_gate_count_d.append(agents[sorted_indices[-2]].num_gates)
            third_best_agent_gate_count_d.append(agents[sorted_indices[-3]].num_gates)

        # for simplicity, convert to numpy array
        top_indices = np.array(top_indices)
        scores = np.array(scores)

        # save avg
        avg = np.mean(scores)
        averages.append(avg)

        # save avg of top agents
        avg_top = np.mean(scores[top_indices])
        averages_top.append(avg_top)

        # save std_dev of scores
        std_dev_score = np.std(scores)

        # std_dev of coin_avg
        coin_total = np.array(coin_total)
        std_dev_coin_total = np.std(coin_total)

        # std_dev of own_coin_avg
        own_coin_total = np.array(own_coin_total)
        std_dev_own_coin_total = np.std(own_coin_total)

        # std_dev of top agents
        std_dev_top = np.std(scores[top_indices])

        if(USE_DYNAMIC):
            # calculate mutation power  and tournament size depending on the current generation
            param_mut_power = interpolate_generations(START_PARAM_MUT_POW, END_PARAM_MUT_POW, generation)
            arch_mut_power = interpolate_generations(START_ARCH_MUT_POW, END_ARCH_MUT_POW, generation)
            TOURNAMENT_SIZE = int(round(interpolate_generations(TOURN_START, TOURN_END, generation)))
        else: # use static mutation power
            param_mut_power = PARAM_MUTATION_POWER
            arch_mut_power = ARCH_MUTATION_POWER

        current_generation_d.append(generation + 1)
        generations_d.append(GENERATIONS)
        test_type.append(f"VQC-{circ_type}-level")
        seed_d.append(seed)
        if SELECTION_TYPE == "tournament":
            selection_type.append(f"{SELECTION_TYPE}: size {TOURNAMENT_SIZE}")
        elif SELECTION_TYPE == "truncation":
            selection_type.append(f"{SELECTION_TYPE}: {NUM_TOP_AGENTS} top agents")
        if evolution_type == "mut-recomb":
            evo_type.append("mutation + recombination")
        elif evolution_type == "mut-only":
            evo_type.append("mutation only")
        population_size_d.append(NUM_AGENTS)
        mutation_rate_d.append(MUTATION_RATE)
        if generation + 1 == GENERATIONS:
            param_mutation_power_d.append("-")
        else:
            param_mutation_power_d.append(param_mut_power)
        if generation + 1 == GENERATIONS or NUM_ARCH_MUT is not None:
            arch_mutation_power_d.append("-")
        else:
            arch_mutation_power_d.append(arch_mut_power)
        if NUM_ARCH_MUT is not None:
            num_arch_mut_d.append(NUM_ARCH_MUT)
        else:
            num_arch_mut_d.append("-")
        top_limit_d.append(NUM_TOP_AGENTS)
        avg_score_d.append(score_avg)
        std_dev_score_d.append(std_dev_score)
        std_dev_coin_total_d.append(std_dev_coin_total)
        std_dev_own_coin_total_d.append(std_dev_own_coin_total)
        coin_avg_d.append(coin_avg)
        own_coin_avg_d.append(own_coin_avg)
        time_d.append(time.time() - start_time)

        # print info
        print(f"Generation {generation + 1} of {GENERATIONS} | Mean score: {score_avg} | Best score: {scores[sorted_indices[-1]]}")
       
        # create dataframe
        data = {
            "current_generation": current_generation_d,
            "num_generations": generations_d,
            "type": test_type,
            "seed": seed_d,
            "sel_type": selection_type,
            "evo_type": evo_type,
            "population_size": population_size_d,
            "mut_rate": mutation_rate_d,
            "param_mut_power": param_mutation_power_d,
            "arch_mut_power": arch_mutation_power_d,
            "num_arch_mut": num_arch_mut_d,
            "top_limit": top_limit_d,
            "avg_score": avg_score_d,
            "std_dev_score": std_dev_score_d,
            "coin_avg": coin_avg_d,
            "std_dev_coin_total": std_dev_coin_total_d,
            "own_coin_avg": own_coin_avg_d,
            "std_dev_own_coin_total": std_dev_own_coin_total_d,
            "time": time_d,
            "best_agent_score": best_agent_score_d,
            "best_agent_coin_total": best_agent_coin_total_d,
            "best_agent_own_coin_total": best_agent_own_coin_total_d,
        }
        if circ_type == "layer":
            data["layer_avg"] = layer_avg_d
            data["best_agent_layer_count"] = best_agent_layer_count_d
        if circ_type == "prototype":
            data["gate_avg_per_layer"] = gate_avg_per_layer_d
            data["layer_num"] = layer_num_d
            data["best_agent_gate_count_per_layer"] = best_agent_gate_count_per_layer_d
        if circ_type == "gate":
            data["gate_avg"] = gate_avg_d
            data["best_agent_gate_count"] = best_agent_gate_count_d
        data["best_agent_parameterized_gates"] = best_agent_parameterized_gates_d
        data["second_best_agent_score"] = second_best_agent_score_d
        data["second_best_agent_coin_total"] = second_best_agent_coin_total_d
        data["second_best_agent_own_coin_total"] = second_best_agent_own_coin_total_d
        data["third_best_agent_score"] = third_best_agent_score_d
        data["third_best_agent_coin_total"] = third_best_agent_coin_total_d
        data["third_best_agent_own_coin_total"] = third_best_agent_own_coin_total_d
        if circ_type == "layer":
            data["second_best_agent_layer_count"] = second_best_agent_layer_count_d
            data["third_best_agent_layer_count"] = third_best_agent_layer_count_d
        if circ_type == "prototype":
            data["second_best_agent_gate_count_per_layer"] = second_best_agent_gate_count_per_layer_d
            data["third_best_agent_gate_count_per_layer"] = third_best_agent_gate_count_per_layer_d
        if circ_type == "gate":
            data["second_best_agent_gate_count"] = second_best_agent_gate_count_d
            data["third_best_agent_gate_count"] = third_best_agent_gate_count_d
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(data)

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        if(generation == 0): 
            print(f"Best agent in first generation:")
            index_of_best_agent = top_indices[-1]
            if agents[index_of_best_agent].circ_type == "layer":
                agents[index_of_best_agent].show(torch.rand((6)))
            else:
                agents[index_of_best_agent].show(torch.rand((6)), agents[index_of_best_agent].gate_list)
            print("\n")
        # create new generation
        if(generation + 1 < GENERATIONS):
            if not isinstance(NUM_ARCH_MUT, int):
                raise ValueError("Number of architecture mutations must be an integer")
            if(evolution_type == "mut-recomb"):
                if not isinstance(MUTATION_RATE, float): 
                    raise ValueError("Mutation rate must be a float")
                children = runner.create_new_gen(agents, scores.tolist(), SELECTION_TYPE, TOURNAMENT_SIZE, NUM_TOP_AGENTS, MUTATION_RATE, param_mut_power, arch_mut_power, NUM_ARCH_MUT)
            elif(evolution_type == "mut-only"):
                children = runner.create_new_gen_mut_only(agents, scores.tolist(), SELECTION_TYPE, TOURNAMENT_SIZE, NUM_TOP_AGENTS, param_mut_power, arch_mut_power, NUM_ARCH_MUT)
            else:
                raise ValueError("invalid evolution type")
            agents = children

            # if exploitation is set to TRUE more exploitation in last quarter of generations (using truncation selection)
            if generation + 1 == int((GENERATIONS-1) / 2) and EXPLOIT:
                SELECTION_TYPE = 'truncation'
                NUM_TOP_AGENTS = 5
        # print moves of best agent in last generation
        else: # last generation
            print(f"Best agent in last generation:")
            index_of_best_agent = top_indices[-1]
            if agents[index_of_best_agent].circ_type == "layer":
                agents[index_of_best_agent].show(torch.rand((6)))
            else:
                agents[index_of_best_agent].show(torch.rand((6)), agents[index_of_best_agent].gate_list)
            """ runner.run_multi_agent(
                env, {"player_1": agents[index_of_best_agent], "player_2": agents[index_of_best_agent]}, seed=seed, render=True
            )
            print("\n") """

    df.to_csv(f"data/Data-{circ_type}-{evolution_type}_{dt_string}.csv")


if __name__ == "__main__":
    s, t, e = parse_args()
    main(s, t, e)
   
