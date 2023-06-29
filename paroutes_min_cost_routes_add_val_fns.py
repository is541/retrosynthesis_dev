"""
Script to compute value of target smiles to compare with actual costs registered
"""
from __future__ import annotations

import pandas as pd
import numpy as np
# import os

# from syntheseus.search.chem import Molecule
# from syntheseus.search.graph.and_or import AndNode
# from syntheseus.search.algorithms.best_first.retro_star import RetroStarSearch, MolIsPurchasableCost
# from syntheseus.search.analysis.solution_time import get_first_solution_time
# from syntheseus.search.analysis.route_extraction import min_cost_routes
# from syntheseus.search.reaction_models.base import BackwardReactionModel
# from syntheseus.search.mol_inventory import BaseMolInventory
# from syntheseus.search.node_evaluation.base import (
#     BaseNodeEvaluator,
#     NoCacheNodeEvaluator,
# )
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from value_functions import initialize_value_functions
from embedding_model import FingerprintModel, GNNModel, load_embedding_model

from paroutes import PaRoutesInventory#, PaRoutesModel, get_target_smiles
# from example_paroutes import PaRoutesRxnCost
# from neighbour_value_functions import (
#     DistanceToCost,
#     TanimotoNNCostEstimator,
#     ConstantMolEvaluator,
# )
from tqdm.auto import tqdm


if __name__ == "__main__":
    # Define input and output folders
    input_folder = 'Runs'
    dataset_str = 'paroutes'
    # dataset_str = 'guacamol'
    
    # Read data
    data_dict = {}
    # for test_db in [file for file in os.listdir(f'Input/{input_folder}') if '.csv' in file]:
    #     test_db_name = test_db.replace('.csv','')
    #     data_dict[test_db_name] = pd.read_csv(f'Input/{input_folder}/{test_db}')
    # for dataset_str in dataset_strs:
    if dataset_str=='paroutes':
        run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"
    elif dataset_str=='guacamol':
        run_id = "Guacamol_combined"
    input_path = f'{input_folder}/{run_id}'
    output_folder = input_path
    data_dict[dataset_str] = pd.read_csv(f'{input_path}/{run_id}_results.csv')
    
    # Define inventory
    inventory=PaRoutesInventory(n=5)
    
    value_fns_names = [
        'constant-0',
        'Tanimoto-distance',
        # 'Tanimoto-distance-TIMES10',
        # 'Tanimoto-distance-TIMES100',
        # 'Tanimoto-distance-EXP',
        # 'Tanimoto-distance-SQRT',
        "Tanimoto-distance-NUM_NEIGHBORS_TO_1",
        "Embedding-from-fingerprints",
        # "Embedding-from-fingerprints-TIMES10",
        # "Embedding-from-fingerprints-TIMES100",
    ]
    fnps_experiment_name = 'fingerprints_v1'
    
    
    device, model_fnps, config_fnps = load_embedding_model(experiment_name=fnps_experiment_name)
    distance_type_fnps = 'cosine'
    value_fns = initialize_value_functions(value_fns_names=value_fns_names, inventory=inventory, model_fnps=model_fnps, distance_type_fnps=distance_type_fnps)
    
    
    # 1. Remove infs
    for name in data_dict.keys(): 
        data_dict[name] = data_dict[name].replace(np.inf, -1)

    # 2. Add features
    # Binned cost (lowest_cost_route_found)
    cost_variable = 'lowest_cost_route_found'
    binned_var_name = 'lowest_cost_route_(binned)'
    num_bins = 20
    lc_bin_ranges_dict = {}
    lc_bin_labels_dict = {}
    for test_db_name, test_data in data_dict.items():
        min_value = 0 # int(test_data['lowest_cost_route_found'].min())
        max_value = int(test_data[cost_variable].max())
        bin_range = np.array([-1, -0.5])
        bin_range = np.append(bin_range, np.linspace(min_value, max_value, num_bins+1, dtype=int))
    #     bin_range = np.linspace(min_value, max_value, num_bins+1, dtype=int)
        bin_labels = [f'{str(int(round(lower,0))).zfill(3)}-{str(int(round(upper,0))).zfill(3)}' for lower, upper in zip(bin_range[:-1], bin_range[1:])]
        bin_labels[0] = 'NotSolved'
        bin_labels[1] = '000'
        lc_bin_ranges_dict[test_db_name] = bin_range
        lc_bin_labels_dict[test_db_name] = bin_labels
        

    # Is purchasable
    purchasable_mols_smiles = [mol.smiles for mol in inventory.purchasable_mols()]

    for name in data_dict.keys(): 
        # Is purchasable
        data_dict[name]['is_purchasable'] = (data_dict[name]['smiles'].isin(purchasable_mols_smiles)) * 1.0
        
        # Binned cost (lowest_cost_route_found)
        data_dict[name][binned_var_name] = pd.cut(data_dict[name][cost_variable], bins=lc_bin_ranges_dict[test_db_name], labels=lc_bin_labels_dict[test_db_name], include_lowest=True)
    #     pd.Series.cat.add_categories(data_dict[name][binned_var_name], ['NotSolved', '000'])
        data_dict[name].loc[data_dict[name][cost_variable] == -1, binned_var_name] = 'NotSolved'
        data_dict[name].loc[data_dict[name][cost_variable] == 0, binned_var_name] = '000'


    for name in data_dict.keys():     
        smiles_list = data_dict[name]['smiles'].unique()
        for value_function_name, value_function in tqdm(value_fns):
            smiles_value_fn_dict = value_function.evaluate_molecules(smiles_list)
            data_dict[name][value_function_name] = data_dict[name]['smiles'].map(smiles_value_fn_dict)
    
    
    column_order = [
        'smiles', 'n_iter', 'first_soln_time', 
        'lowest_cost_route_found', 'lowest_cost_route_(binned)',
        'best_route_cost_lower_bound', 'lowest_depth_route_found',
        'best_route_depth_lower_bound', 'num_calls_rxn_model',
        'num_nodes_in_tree', 
        'is_purchasable',
        # 'constant-0', 'Tanimoto-distance',
        # 'Tanimoto-distance-TIMES10', 
        # 'Tanimoto-distance-EXP',
        # 'Tanimoto-distance-SQRT', 
        # 'Tanimoto-distance-NUM_NEIGHBORS_TO_1',
        # 'Tanimoto-distance-NUM_NEIGHBORS_TO_1_TIMES1000',
    ]
    column_order = column_order + value_fns_names

    for test_db_name, test_data in data_dict.items(): 
        test_data = test_data[column_order]
        test_data.to_csv(f'{output_folder}/{test_db_name}_result_added_value_fns.csv', index=False)
    
