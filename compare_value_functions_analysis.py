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
import time
import plotly.graph_objects as go

def plot_result(df_result, image_suffix, labelalias):
    df_solved = df_result[df_result['not_solved'] == 0].groupby('algorithm_alias').size().reset_index(name='num_mol_solved')
    df_not_solved = df_result[df_result['not_solved'] == 1].groupby('algorithm_alias').size().reset_index(name='num_mol_not_solved')

    algs_considered = df_result['algorithm'].unique()
    labelalias_considered = [labelalias[name] for name in algs_considered]
    categoryarray_list = [label for label in labelalias.values() if label in labelalias_considered]
    
    fig = px.box(df_result, x="algorithm_alias", y="value", width=1000, height=600,
                color='not_solved',
                color_discrete_sequence=[px.colors.qualitative.Plotly[1], px.colors.qualitative.Plotly[0]],
                boxmode="overlay", 
                points='all',
                labels={
                    "value": "Time to first solution",
                },
                title="Time to first solution using different value functions",
                category_orders={"algorithm_alias": categoryarray_list},
            )
    fig.update_layout(xaxis_title=None)

    # fig.update_layout(xaxis_title=None)
    # fig.update_layout(xaxis=dict(
    #     # tickmode='array',
    #     # tickvals=categoryarray_list,
    #     # ticktext=categoryarray_list, #list(labelalias),
    #     categoryorder='array',
    #     categoryarray=categoryarray_list
    # ))
    # fig.update_xaxes(labelalias=labelalias, categoryorder='array', categoryarray=categoryarray_list)

    # Add lines above each box
    for _, row in df_solved.iterrows():
        algorithm = row['algorithm_alias']
        num_mol_solved = row['num_mol_solved']
        num_mol_not_solved = df_not_solved[df_not_solved['algorithm_alias'] == algorithm]['num_mol_not_solved'].item()
        box_y_1 = max_value*1.32# df_result[df_result['algorithm_alias'] == algorithm]['value'].max()
        box_y_2 = max_value*1.28# df_result[df_result['algorithm_alias'] == algorithm]['value'].max()

        fig.add_annotation(
            x=algorithm, y=box_y_2, text=f"mol solved: {num_mol_solved}",
            showarrow=False, font=dict(color=px.colors.qualitative.Plotly[0])
        )
        
        fig.add_annotation(
            x=algorithm, y=box_y_1, text=f"mol not solved: {num_mol_not_solved}",
            showarrow=False, font=dict(color=px.colors.qualitative.Plotly[1])
        )

    fig.write_image(f'{output_folder}/Boxplot_time_first_solution_{image_suffix}.pdf') 
    time.sleep(10)
    fig.write_image(f'{output_folder}/Boxplot_time_first_solution_{image_suffix}.pdf')
    
    
    # PLOT INVERTING COLORS
    fig = px.box(df_result, x="algorithm_alias", y="value", width=1000, height=600,
                color='not_solved',
                color_discrete_sequence=[px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]],
                boxmode="overlay", 
                points='all',
                labels={
                    "value": "Time to first solution",
                },
                title="Time to first solution using different value functions",
                category_orders={"algorithm_alias": categoryarray_list},
            )
    fig.update_layout(xaxis_title=None)


    # Add lines above each box
    for _, row in df_solved.iterrows():
        algorithm = row['algorithm_alias']
        num_mol_solved = row['num_mol_solved']
        num_mol_not_solved = df_not_solved[df_not_solved['algorithm_alias'] == algorithm]['num_mol_not_solved'].item()
        box_y_1 = max_value*1.32# df_result[df_result['algorithm_alias'] == algorithm]['value'].max()
        box_y_2 = max_value*1.28# df_result[df_result['algorithm_alias'] == algorithm]['value'].max()

        fig.add_annotation(
            x=algorithm, y=box_y_2, text=f"mol solved: {num_mol_solved}",
            showarrow=False, font=dict(color=px.colors.qualitative.Plotly[0])
        )
        
        fig.add_annotation(
            x=algorithm, y=box_y_1, text=f"mol not solved: {num_mol_not_solved}",
            showarrow=False, font=dict(color=px.colors.qualitative.Plotly[1])
        )
    
    fig.write_image(f'{output_folder}/Boxplot_time_first_solution_col_v2_{image_suffix}.pdf') 
    time.sleep(10)
    fig.write_image(f'{output_folder}/Boxplot_time_first_solution_col_v2_{image_suffix}.pdf')
    



def create_result_df(result, name):
    assert name == result[name].name, f"name: {name} is different from result[name].name: {result[name].name}"
    
    soln_time_dict = result[name].soln_time_dict
    # num_different_routes_dict = result[name].num_different_routes_dict
    # final_num_rxn_model_calls_dict = result[name].final_num_rxn_model_calls_dict
    # final_num_value_function_calls_dict = result[name].final_num_value_function_calls_dict
    # output_graph_dict = result[name].output_graph_dict
    # routes_dict = result[name].routes_dict

    # df_results = pd.DataFrame()
    df_soln_time = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})
    # df_different_routes = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})
    print("Create dataframe solution times")
    #     for name_alg, value_dict  in soln_time_dict.items():
    for smiles, value  in soln_time_dict.items():
        row_soln_time = {'algorithm': name, 'similes': smiles, 'property':'sol_time', 'value': value}

        df_soln_time = pd.concat([df_soln_time, pd.DataFrame([row_soln_time])], ignore_index=True)

    # #     for name_alg, value_dict  in num_different_routes_dict.items():
    # for smiles, value  in num_different_routes_dict.items():
    #     row_different_routes = {'algorithm': name, 'similes': smiles, 'property':'diff_routes', 'value': value}

    #     df_different_routes = pd.concat([df_different_routes, pd.DataFrame([row_different_routes])], ignore_index=True)

    # df_results_tot = pd.concat([df_soln_time, df_different_routes], axis=0)
    return df_soln_time


if __name__ == "__main__":
    # eventid = '202306-3017-5132-d4cd54bf-e5a4-44c5-82af-c629a3692d87_HARDEST'
    # eventid = '202307-0320-4900-d4c27728-a5aa-4177-8681-268a17c3d208_HARD'
    # eventid = '202307-0620-3725-cc7b1f07-14cd-47e8-9d40-f5b2f358fa28_MID_HARD'
    # eventid = 'MID'
    # eventid = 'MID_EASY'
    eventid = 'MID_EASY_COST1'
    plot_existing_results = False
    
    output_folder = f"CompareTanimotoLearnt/{eventid}"
    
    
    algs_to_consider = 'all'
    # algs_to_consider = [
    #     # 'constant-0',
    #     # 'Tanimoto-distance',
    #     # 'Tanimoto-distance-TIMES01',
    #     # 'Tanimoto-distance-TIMES03',
    #     # 'Tanimoto-distance-TIMES10',
    #     # 'Tanimoto-distance-TIMES100',
    #     # 'Tanimoto-distance-EXP',
    #     # 'Tanimoto-distance-SQRT',
    #     # "Tanimoto-distance-NUM_NEIGHBORS_TO_1",
    #     "Embedding-from-fingerprints",
    #     # "Embedding-from-fingerprints-TIMES10",
    #     # "Embedding-from-fingerprints-TIMES100",
    #     "Embedding-from-gnn",
    # ]
    # algs_considered = []
    
    if plot_existing_results:
        df_results_tot = pd.read_csv(f'{output_folder}/results_all.csv')
    else:
        result = {}
        for file_name in tqdm([file for file in os.listdir(output_folder) if 'pickle' in file]):
            name = file_name.replace('.pickle','').replace('result_','')
            print(name)
            if (algs_to_consider == 'all') | (name in algs_to_consider):
                with open(f'{output_folder}/{file_name}', 'rb') as handle:
                    result[name] = pickle.load(handle)
                # algs_considered.append(name)
        print("Loaded algorithm pickle")
        df_results_tot = pd.DataFrame({'algorithm': [], 'similes': [], 'property':[], 'value': []})
        
        for name in tqdm(result.keys()):
            df_results_alg = create_result_df(result, name)
            df_results_tot = pd.concat([df_results_tot, df_results_alg], axis=0)
            
        print("Save dataframe to csv")
        df_results_tot.to_csv(f'{output_folder}/results_all.csv', index=False)
        print("Saved dataframe to csv")
        
    
    # if algs_to_consider != 'all':
    #     df_results_tot = df_results_tot.loc[df_results_tot['algorithm'].isin(algs_to_consider)]

    # Solution time
    # print("Create plot")
    results_solution_times = df_results_tot.loc[df_results_tot['property']=='sol_time']
    df_result = results_solution_times.copy()
    df_result["algorithm_alias"] = df_result["algorithm"].map(labelalias)
    
    # Deal with unsolved molecules 
    df_result["not_solved"] = (df_result['value'] == np.inf) * 1
    max_value = df_result[df_result['value'] != np.inf]['value'].max()
    df_result.loc[df_result['value'] == np.inf, 'value'] = 1.2 * max_value
    
    df_results_grouped = df_result.groupby(["algorithm"], as_index=False).agg(nr_mol_not_solved=pd.NamedAgg(column="not_solved", aggfunc="sum"))
    df_results_grouped.to_csv(f'{output_folder}/num_mol_not_solved.csv', index=False)

    
    # Plot
    # breakpoint()
    plot_result(df_result=df_result, image_suffix="all", labelalias=labelalias)
    
    df_result_Tanimoto = df_result.loc[(df_result["algorithm"]=='constant-0') | (df_result["algorithm"].str.contains('Tanimoto'))]
    plot_result(df_result=df_result_Tanimoto, image_suffix="Tanimoto", labelalias=labelalias)
    
    df_result_Embedding = df_result.loc[(df_result["algorithm"]=='constant-0') | (df_result["algorithm"].str.contains('Embedding'))]
    plot_result(df_result=df_result_Embedding, image_suffix="Embedding", labelalias=labelalias)
    
    