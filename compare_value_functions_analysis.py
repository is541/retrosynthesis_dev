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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def sort_names_according_to_labelalias_key_order(list_names, labelalias):
    labelalias_order = labelalias.keys()
    index_dict = {name: index for index, name in enumerate(labelalias_order)}

    # Sort list_names based on their index in sorted_list
    sorted_names = sorted(list_names, key=lambda x: index_dict[x], reverse=True)
    
    return sorted_names

def sort_names_according_to_labelalias_value_order(list_names, labelalias):
    labelalias_order = labelalias.values()
    # print(labelalias_order)
    index_dict = {name: index for index, name in enumerate(labelalias_order)}

    # Sort list_names based on their index in sorted_list
    sorted_names = sorted(list_names, key=lambda x: index_dict[x], reverse=True)
    # print(sorted_names)
    
    return sorted_names

def create_quantiles_df(data):
    grouped_data = data.groupby(['algorithm_alias', 'property'])

    # Define the quantiles to compute
    # quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

    # Function to compute quantiles for a given group
    def compute_quantiles(group):
        return group['value'].quantile(quantiles)

    # Compute quantiles for each group
    result = grouped_data.apply(compute_quantiles)
    return result

def plot_quantiles_df(df, column_order):
    # Transpose the DataFrame and drop the 'property' column
    df_transposed = df.drop(columns=['property']).set_index('algorithm_alias').T
    df_transposed = df_transposed[column_order]
    df_transposed = df_transposed.iloc[:, ::-1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot the table with color-coded cells (using YlOrRd colormap)
    sns.heatmap(df_transposed, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax,
                annot_kws={'weight': 'normal'})

    # Remove ticks from both x and y axes
    ax.tick_params(bottom=False, left=False)
    font_properties = FontProperties(weight='bold')
    plt.xticks(fontproperties=font_properties)
    plt.yticks(fontproperties=font_properties)

    # Rotate y-axis labels to be horizontal
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.5)

    # Get the colorbar and set a new label
    cbar = ax.collections[0].colorbar
    cbar.set_label('')

    # Add title and axis labels
    plt.title('Solution Time for different percentiles of solved molecules')
    plt.xlabel('Algorithm')
    plt.ylabel('Solved Molecules Percentile')

    plt.tight_layout()

    # Save the plot to a PDF file
    fig = plt.gcf()
    return fig

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
        df_not_solved_algorithm = df_not_solved[df_not_solved['algorithm_alias'] == algorithm]
        if df_not_solved_algorithm.shape[0] > 0:
            num_mol_not_solved = df_not_solved_algorithm['num_mol_not_solved'].item()
        else:
            num_mol_not_solved = 0
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
        df_not_solved_algorithm = df_not_solved[df_not_solved['algorithm_alias'] == algorithm]
        if df_not_solved_algorithm.shape[0] > 0:
            num_mol_not_solved = df_not_solved_algorithm['num_mol_not_solved'].item()
        else:
            num_mol_not_solved = 0
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
    # eventid = 'MID_EASY_COST1'
    # eventid = 'MID_EASY_COST1_ALL'
    # eventid = 'MID_EASY_v2_cost_paroutes'
    eventid = 'MID_EASY_v3'
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
    
    # Create quantiles df
    # Group the data by 'algorithm' and 'property'
    result_quantiles = create_quantiles_df(df_result)
    result_quantiles = result_quantiles.reset_index()
    result_quantiles.to_csv(f'{output_folder}/results_quantiles.csv', index=False)
    
    column_order = result_quantiles['algorithm_alias'].unique()
    fig_quantiles = plot_quantiles_df(df=result_quantiles, column_order=column_order)
    fig_quantiles.savefig(f'{output_folder}/quantiles_table_all.pdf', format='pdf')

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
    
    df_result_quantiles_Tanimoto = result_quantiles.loc[(result_quantiles["algorithm_alias"]=='constant-0') | (result_quantiles["algorithm_alias"].str.contains('Tanimoto'))]
    column_order_Tanimoto = df_result_quantiles_Tanimoto['algorithm_alias'].unique()
    column_order_Tanimoto = sort_names_according_to_labelalias_value_order(column_order_Tanimoto, labelalias)
    fig_quantiles_Tanimoto = plot_quantiles_df(df=df_result_quantiles_Tanimoto, column_order=column_order_Tanimoto)
    fig_quantiles_Tanimoto.savefig(f'{output_folder}/quantiles_table_Tanimoto.pdf', format='pdf')
    
    df_result_Embedding = df_result.loc[(df_result["algorithm"]=='constant-0') | (df_result["algorithm"].str.contains('Embedding'))]
    plot_result(df_result=df_result_Embedding, image_suffix="Embedding", labelalias=labelalias)
    
    df_result_quantiles_Embedding = result_quantiles.loc[(result_quantiles["algorithm_alias"]=='constant-0') | (result_quantiles["algorithm_alias"].str.contains('Embedding'))]
    column_order_Embedding = df_result_quantiles_Embedding['algorithm_alias'].unique()
    column_order_Embedding = sort_names_according_to_labelalias_value_order(column_order_Embedding, labelalias)
    fig_quantiles_Embedding = plot_quantiles_df(df=df_result_quantiles_Embedding, column_order=column_order_Embedding)
    fig_quantiles_Embedding.savefig(f'{output_folder}/quantiles_table_Embedding.pdf', format='pdf')
    
    