"""Plotting script for visualizing the results of the VQC experiments."""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime

def read_csv_files(folder_path, names, index):
    print(f"Reading CSV files for {names[index]} from: {os.path.abspath(folder_path)}")
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def calculate_avg(data, plot_type):
    if plot_type == 'score':
        avg_scores = data.groupby('current_generation')['avg_score'].agg(['mean', 'std']).reset_index()
        return avg_scores
    elif plot_type == 'own_coins':
        avg_own_coins = data.groupby('current_generation')['own_coin_avg'].agg(['mean', 'std']).reset_index()
        return avg_own_coins
    elif plot_type == 'own_coin_rate':
        data['own_coin_rate_avg'] = data['own_coin_avg'] / data['coin_avg']
        avg_own_coin_rate = data.groupby('current_generation')['own_coin_rate_avg'].agg(['mean', 'std']).reset_index()
        return avg_own_coin_rate
    elif plot_type == 'total_coins':
        avg_total_coins = data.groupby('current_generation')['coin_avg'].agg(['mean', 'std']).reset_index()
        return avg_total_coins

def calculate_best_agent(data, plot_type):
    if plot_type == 'score':
        best_agent_scores = data.groupby('current_generation')['best_agent_score'].agg(['mean', 'std']).reset_index()
        return best_agent_scores
    elif plot_type == 'own_coins':
        best_agent_own_coins = data.groupby('current_generation')['best_agent_own_coin_total'].agg(['mean', 'std']).reset_index()
        return best_agent_own_coins
    elif plot_type == 'own_coin_rate':
        data['best_agent_own_coin_rate'] = data['best_agent_own_coin_total'] / data['best_agent_coin_total']
        best_agent_own_coin_rate = data.groupby('current_generation')['best_agent_own_coin_rate'].agg(['mean', 'std']).reset_index()
        return best_agent_own_coin_rate
    elif plot_type == 'total_coins':
        best_agent_total_coins = data.groupby('current_generation')['best_agent_coin_total'].agg(['mean', 'std']).reset_index()
        return best_agent_total_coins

def calculate_gate_total(data):
    if data['type'].iloc[0] == 'VQC-layer-level':
        data['best_agent_gate_count'] = data['best_agent_layer_count'] * 6 * 4
    elif data['type'].iloc[0] == 'VQC-prototype-level':
        data['best_agent_gate_count'] = data['layer_num'] * data['best_agent_gate_count_per_layer']
    best_agent_gate_count = data.groupby('current_generation')['best_agent_gate_count'].agg(['mean', 'std']).reset_index()
    return best_agent_gate_count

def calculate_gate_param(data):
    best_agent_parameterized_gates = data.groupby('current_generation')['best_agent_parameterized_gates'].agg(['mean', 'std']).reset_index()
    return best_agent_parameterized_gates

def apply_ewm(data, span=10):
    data['mean_ewm'] = data['mean'].ewm(span=span).mean()
    data['std_ewm'] = data['std'].ewm(span=span).mean()
    return data

def plot_data(first_list, second_list, names, output_path, plot_type):
    sns.set_context('paper', font_scale=1.5)  # use context 'talk' for presentation slides?
    sns.set_style('whitegrid') 

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14
    })

    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("colorblind", len(names))

    # use this to specify a certain color for a specific line
    """ colors2 = sns.color_palette("colorblind", 3)
    orange = colors2[1] """

    # for plot_type gate: first: parameterized gates, second: total gates
    # for other plot_types: first: avg, second: best_agent
    for first, second, name, color in zip(first_list, second_list, names, colors):
        # Plot avg values / total gate count with standard deviation
        linestyle = '--'
        if plot_type == 'gates': linestyle = ':'
        sns.lineplot(data=first, x='current_generation', y='mean_ewm', color=color, linestyle=linestyle, linewidth=2.5)
        plt.fill_between(first['current_generation'], first['mean_ewm'] - first['std_ewm'], first['mean_ewm'] + first['std_ewm'], color=color, alpha=0.1)
        
        sns.lineplot(data=second, x='current_generation', y='mean_ewm', color=color, linestyle='-', linewidth=2.5)
        plt.fill_between(second['current_generation'], second['mean_ewm'] - second['std_ewm'], second['mean_ewm'] + second['std_ewm'], color=color, alpha=0.1)

    plt.xlabel('Generation', fontsize=16, fontfamily='DejaVu Serif')
    if plot_type == 'score':
        plt.ylabel('Score', fontsize=16, fontfamily='DejaVu Serif')
    elif plot_type == 'own_coins':
        plt.ylabel('Own Coins', fontsize=16, fontfamily='DejaVu Serif')
    elif plot_type == 'own_coin_rate':
        plt.ylabel('Own Coin Rate', fontsize=16, fontfamily='DejaVu Serif')
    elif plot_type == 'total_coins':
        plt.ylabel('Total Coins', fontsize=16, fontfamily='DejaVu Serif')
    elif plot_type == 'gates':
        plt.ylabel('Gate Count', fontsize=16, fontfamily='DejaVu Serif')
    
    # Custom legend
    handles = []
    labels = []

    # Add Approach section
    handles.append(Line2D([0], [0], color='w', label='Approach', linestyle=''))
    labels.append('Approach')
    for name, color in zip(names, colors):
        handles.append(Line2D([0], [0], color=color, lw=2))
        labels.append(name)

    # Add Data Type section
    if plot_type == 'gates':
        handles.append(Line2D([0], [0], color='w', label='Gate Metrics', linestyle=''))
        labels.append('Gate Metrics')
        handles.append(Line2D([0], [0], color='k', linestyle='-', lw=2))
        labels.append('Total')
        handles.append(Line2D([0], [0], color='k', linestyle=':', lw=2))
        labels.append('Parameterized')
    else:
        handles.append(Line2D([0], [0], color='w', label='Data Type', linestyle=''))
        labels.append('Data Type')
        handles.append(Line2D([0], [0], color='k', linestyle='-', lw=2))
        labels.append('Best')
        handles.append(Line2D([0], [0], color='k', linestyle='--', lw=2))
        labels.append('Average')

    plt.legend(handles=handles, labels=labels, loc='lower center') # choose fitting location
    plt.tight_layout()
    plt.savefig(output_path) 

def main(folder_paths, names, output_directory, output_name, plot_type):
    current_working_directory = os.getcwd()    
    # Check if in the correct directory (src)
    if os.path.basename(current_working_directory) != "src":
        print("Error: The script must be run from the 'src' directory.")
        sys.exit(1)
    
    avg_list = []
    best_agent_list = []

    total_gates_list = []
    param_gates_list = []
    
    for index, folder_path in enumerate(folder_paths):
        data = read_csv_files(folder_path, names, index)
        if plot_type == 'gates':
            total_gates = calculate_gate_total(data)
            param_gates = calculate_gate_param(data)
            total_gates_ewm = apply_ewm(total_gates)
            param_gates_ewm = apply_ewm(param_gates)
            total_gates_list.append(total_gates_ewm)
            param_gates_list.append(param_gates_ewm)
        else:
            avg = calculate_avg(data, plot_type)
            best_agent = calculate_best_agent(data, plot_type)
            avg_ewm = apply_ewm(avg) 
            best_agent_ewm = apply_ewm(best_agent)
            avg_list.append(avg_ewm)
            best_agent_list.append(best_agent_ewm)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_directory, f'{output_name}_{current_time}.png')
    
    if plot_type == 'gates':
        plot_data(param_gates_list, total_gates_list, names, output_path, plot_type)
    else:
        plot_data(avg_list, best_agent_list, names, output_path, plot_type)

if __name__ == '__main__':
    ## HOW TO PLOT DATA
    # 1. Define the plot type 
        # e.g. plot_type = 'score'
    # 2. Define the folder paths to the CSV files
        # e.g. folder_paths = ['../data/1-Recomb/Mu', '../data/1-Recomb/ReMu']
    # 3. Define the names of the models
        # e.g. names = ['Mu', 'ReMu']

    # POSSIBLE PLOT TYPES
    #plot_type = 'score'
    #plot_type = 'own_coins'
    #plot_type = 'own_coin_rate'
    #plot_type = 'total_coins'
    #plot_type = 'gates'

    # DATA I PLOTTED
    # 1-Recomb Score
    """ plot_type = 'score'
    folder_paths = [
        '../data/RESULTS/1-Recomb/Mu',  # path to first folder containing CSV files
        '../data/RESULTS/1-Recomb/ReMu'  # path to second folder containing CSV files
    ]
    names = ['Gate-Based VQC(70G): Mu', 'Gate-Based VQC(70G): ReMu']  # names corresponding to each folder
    output_name = '1-score'
    output_directory = '../data/RESULTS/1-Recomb'  # path to output directory """

    # 2-Strats Score
    """ plot_type = 'score'
    folder_paths = [
        '../data/RESULTS/2-Strats/Layer',
        '../data/RESULTS/2-Strats/Gate',
        '../data/RESULTS/2-Strats/Prototype'
    ]
    names = ['Layer-Based VQC(1L)', 'Gate-Based VQC(70G)', 'Prototype-Based VQC(8L-18G)']
    output_name = '2-score'
    output_directory = '../data/RESULTS/2-Strats' """

    # 2-Strats Own Coins
    """ plot_type = 'own_coins'
    folder_paths = [
        '../data/RESULTS/2-Strats/Layer',
        '../data/RESULTS/2-Strats/Gate',
        '../data/RESULTS/2-Strats/Prototype'
    ]
    names = ['Layer-Based VQC(1L)', 'Gate-Based VQC(70G)', 'Prototype-Based VQC(8L-18G)']
    output_name = '2-own-coins'
    output_directory = '../data/RESULTS/2-Strats' """

    # 2-Strats Own Coin Rate
    """ plot_type = 'own_coin_rate'
    folder_paths = [
        '../data/RESULTS/2-Strats/Layer',
        '../data/RESULTS/2-Strats/Gate',
        '../data/RESULTS/2-Strats/Prototype'
    ]
    names = ['Layer-Based VQC(1L)', 'Gate-Based VQC(70G)', 'Prototype-Based VQC(8L-18G)']
    output_name = '2-own-coin-rate'
    output_directory = '../data/RESULTS/2-Strats' """

    # 2-Strats Total Coins
    """ plot_type = 'total_coins'
    folder_paths = [
        '../data/RESULTS/2-Strats/Layer',
        '../data/RESULTS/2-Strats/Gate',
        '../data/RESULTS/2-Strats/Prototype'
    ]
    names = ['Layer-Based VQC(1L)', 'Gate-Based VQC(70G)', 'Prototype-Based VQC(8L-18G)']
    output_name = '2-total-coins'
    output_directory = '../data/RESULTS/2-Strats' """

    # 2-Strats Best Agent Gate Count
    """ plot_type = 'gates'
    folder_paths = [
        '../data/RESULTS/2-Strats/Layer',
        '../data/RESULTS/2-Strats/Gate',
        '../data/RESULTS/2-Strats/Prototype'
    ]
    names = ['Layer-Based VQC(1L)', 'Gate-Based VQC(70G)', 'Prototype-Based VQC(8L-18G)']
    output_name = '2-gate-count'
    output_directory = '../data/RESULTS/2-Strats' """

    # 3-Bench Score | Gate 70
    plot_type = 'score'
    folder_paths = [
        '../data/RESULTS/3-Bench/Baseline',  
        '../data/RESULTS/3-Bench/Gate70'  
    ]
    names = ['Static VQC(8L)', 'Gate-Based VQC(70G)'] 
    output_name = '3-70-score'
    output_directory = '../data/RESULTS/3-Bench'  

    # 3-Bench Gate | Gate 70
    """ plot_type = 'gates'
    folder_paths = [
        '../data/RESULTS/3-Bench/Baseline',  
        '../data/RESULTS/3-Bench/Gate70'  
    ]
    names = ['Static VQC(8L)', 'Gate-Based VQC(70G)'] 
    output_name = '3-70-gate-count'
    output_directory = '../data/RESULTS/3-Bench' """ 

    # 3-Bench Gate | Gate 70 and Gate 10
    """ plot_type = 'gates'
    folder_paths = [
        '../data/RESULTS/3-Bench/Baseline',  
        '../data/RESULTS/3-Bench/Gate70',
        '../data/RESULTS/3-Bench/Gate10'  
    ]
    names = ['Static VQC(8L)', 'Gate-Based VQC(70G)', 'Gate-Based VQC(10G)'] 
    output_name = '3-70and50-gate-count'
    output_directory = '../data/RESULTS/3-Bench' """   

    # 3-Bench Score | Gate 10
    """ plot_type = 'score'
    folder_paths = [
        '../data/RESULTS/3-Bench/Baseline', 
        '../data/RESULTS/3-Bench/Gate10'  
    ]
    names = ['Static VQC(8L)', 'Gate-Based VQC(10G)'] 
    output_name = '3-10-score'
    output_directory = '../data/RESULTS/3-Bench' """ 

    # 3-Bench Gate | Gate 10
    """ plot_type = 'gates'
    folder_paths = [
        '../data/RESULTS/3-Bench/Baseline',
        '../data/RESULTS/3-Bench/Gate10'  
    ]
    names = ['Static VQC(8L)', 'Gate-Based VQC(10G)'] 
    output_name = '3-10-gate-count'
    output_directory = '../data/RESULTS/3-Bench'   """

    
    main(folder_paths, names, output_directory, output_name, plot_type)