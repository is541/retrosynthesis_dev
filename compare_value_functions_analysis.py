"""
Script to find minimum cost routes for PaRoutes benchmark
on a list of SMILES provided by the user.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from value_functions import labelalias
from compare_value_functions_run import SearchResult

import pickle
import os
import plotly.express as px
from tqdm.auto import tqdm



def create_result_df(result, name):
    assert name == result[name].name, f"name: {name} is different from result[name].name: {result[name].name}"
    
    soln_time_dict = result[name].soln_time_dict
    num_different_routes_dict = result[name].num_different_routes_dict
    final_num_rxn_model_calls_dict = result[name].final_num_rxn_model_calls_dict
    final_num_value_function_calls_dict = result[name].final_num_value_function_calls_dict
    output_graph_dict = result[name].output_graph_dict
    routes_dict = result[name].routes_dict

    # df_results = pd.DataFrame()
    df_soln_time = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})
    df_different_routes = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})

    #     for name_alg, value_dict  in soln_time_dict.items():
    for smiles, value  in soln_time_dict.items():
        row_soln_time = {'algorithm': name, 'similes': smiles, 'property':'sol_time', 'value': value}

        df_soln_time = pd.concat([df_soln_time, pd.DataFrame([row_soln_time])], ignore_index=True)

    #     for name_alg, value_dict  in num_different_routes_dict.items():
    for smiles, value  in num_different_routes_dict.items():
        row_different_routes = {'algorithm': name, 'similes': smiles, 'property':'diff_routes', 'value': value}

        df_different_routes = pd.concat([df_different_routes, pd.DataFrame([row_different_routes])], ignore_index=True)

    df_results_tot = pd.concat([df_soln_time, df_different_routes], axis=0)
    return df_results_tot


if __name__ == "__main__":
    # eventid = '202306-3017-5132-d4cd54bf-e5a4-44c5-82af-c629a3692d87_HARDEST'
    eventid = '202307-0320-4900-d4c27728-a5aa-4177-8681-268a17c3d208_HARD'
    
    output_folder = f"CompareTanimotoLearnt/{eventid}"
    
    algs_to_consider = 'all'
    # algs_to_consider = [
    #     'constant-0',
    #     'Tanimoto-distance',
    #     'Tanimoto-distance-TIMES10',
    #     'Tanimoto-distance-TIMES100',
    #     'Tanimoto-distance-EXP',
    #     'Tanimoto-distance-SQRT',
    #     "Tanimoto-distance-NUM_NEIGHBORS_TO_1",
    #     "Embedding-from-fingerprints",
    #     "Embedding-from-fingerprints-TIMES10",
    #     "Embedding-from-fingerprints-TIMES100",
    # ]

    result = {}
    for file_name in tqdm([file for file in os.listdir(output_folder) if 'pickle' in file]):
        name = file_name.replace('.pickle','').replace('result_','')
        print(name)
        if (algs_to_consider == 'all') | (name in algs_to_consider):
            with open(f'{output_folder}/{file_name}', 'rb') as handle:
                result[name] = pickle.load(handle)
    
    df_results_tot = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})
    
    for name in tqdm(result.keys()):
        df_results_alg = create_result_df(result, name)
        df_results_tot = pd.concat([df_results_tot, df_results_alg], axis=0)
    df_results_tot.to_csv(f'{output_folder}/results_all.csv', index=False)
    
    # if algs_to_consider != 'all':
    #     df_results_tot = df_results_tot.loc[df_results_tot['algorithm'].isin(algs_to_consider)]

    # Solution time
    results_solution_times = df_results_tot.loc[df_results_tot['property']=='sol_time']
    df_result = results_solution_times.copy()
    
    # Deal with unsolved molecules
    df_result["value_is_inf"] = (df_result['value'] == np.inf) * 1
    max_value = df_result[df_result['value'] != np.inf]['value'].max()
    df_result.loc[df_result['value'] == np.inf, 'value'] = 1.2 * max_value
    
    df_results_grouped = df_result.groupby(["algorithm"], as_index=False).agg(nr_mol_not_solved=pd.NamedAgg(column="value_is_inf", aggfunc="sum"))

    df_results_grouped.to_csv(f'{output_folder}/num_mol_not_solved.csv', index=False)
    
    # Plot
    fig = px.box(df_result, x="algorithm", y="value", width=1000, height=600,
                labels={
    #                      "algorithm": None,
                        "value": "Time to first solution",
    #                      "species": "Species of Iris"
                    },
    #              title="Time to first solution"
                )
    fig.update_layout(xaxis_title=None)
    fig.update_xaxes(labelalias=labelalias, categoryorder='array', categoryarray=list(labelalias.keys()))
    fig.write_image(f'{output_folder}/Boxplot_time_first_solution.png') 




