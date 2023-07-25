import math
import random
import plotly.graph_objects as go
# import plotly.offline as pyo
# pyo.init_notebook_mode()
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
import plotly.express as px
import os
import time


def concatenate_target_routes_data(targ_routes_dict, distances_dict):
    first = True
    for target, data_target in targ_routes_dict.items():
        if len(data_target.keys())>0:
            data_target_route_1 = data_target['route_1'][['smiles', 'depth']]
            data_target_route_1= data_target_route_1.rename(columns={'depth': 'depth_route_1'})
        else:
            data_target_route_1 = pd.DataFrame(columns=['smiles', 'depth_route_1'])
        if len(data_target.keys())>1:
            data_target_route_2 = data_target['route_2'][['smiles',  'depth']]
            data_target_route_2= data_target_route_2.rename(columns={'depth': 'depth_route_2'})
        else:
            data_target_route_2 = pd.DataFrame(columns=['smiles', 'depth_route_2'])
        if len(data_target.keys())>2:
            data_target_route_3 = data_target['route_3'][['smiles',  'depth']]
            data_target_route_3 = data_target_route_3.rename(columns={'depth': 'depth_route_3'})
        else:
            data_target_route_3 = pd.DataFrame(columns=['smiles', 'depth_route_3'])

        data_current_target = pd.merge(data_target_route_1, data_target_route_2, how='outer', on='smiles')
        data_current_target = pd.merge(data_current_target, data_target_route_3, how='outer', on='smiles')
        data_current_target = data_current_target.loc[data_current_target['smiles']!=target]

        # Add distance info
        data_current_target = pd.merge(data_current_target, distances_dict[target], on='smiles', how='left')
        # Add target info
        data_current_target['target'] = target 

        if first:
            data_all_target = data_current_target
            first = False
        else:
            data_all_target = pd.concat([data_all_target, data_current_target])
            
    return data_all_target


def num_heavy_atoms(mol):
#     return Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol)
    return Chem.rdchem.Mol.GetNumAtoms(mol, onlyExplicit=True)


if __name__ == "__main__":
    # distance_to_consider = 'Tanimoto'
    distance_to_consider = 'Fnps-nn'
    # distance_to_consider = 'Gnn'

    if distance_to_consider=='Tanimoto':
    #     distance_col = "Tanimoto_distance_to_target"
    #     rank_col = 'distance_to_target_rank'
        rank_col = 'Tanimoto_distance_to_target_rank'
    elif distance_to_consider=='Fnps-nn':
        rank_col = 'Fnps_distance_to_target_rank'
    elif distance_to_consider=='Gnn':
        rank_col = 'Gnn_distance_to_target_rank'

    run_id = '202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7'
    # input_file = f'Runs/{run_id}/routes_df.csv'
    # routes_df = pd.read_csv(input_file)

    input_file_routes = f'Runs/{run_id}/targ_routes.pickle'
    input_file_distances = f'Runs/{run_id}/targ_to_purch_distances_v2.pickle'

    with open(input_file_routes, 'rb') as handle:
        targ_routes_dict = pickle.load(handle)
        
    # Load distances data
    with open(input_file_distances, 'rb') as handle:
        distances_dict = pickle.load(handle)



    output_folder = f'Plots/{run_id}/Histograms_v2/{distance_to_consider}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Concat routes
    data_all_target = concatenate_target_routes_data(
        targ_routes_dict=targ_routes_dict, 
        distances_dict=distances_dict
    )

    # Molecule complexity
    unique_smiles = data_all_target['smiles'].unique()
    smiles_complexity_dict = {smiles: num_heavy_atoms(Chem.MolFromSmiles(smiles)) for smiles in unique_smiles}

    data_all_target['smiles_nr_explicit_atoms'] = data_all_target['smiles'].map(smiles_complexity_dict)

    # Number explicit atoms
    fig = px.histogram(data_all_target.dropna(subset=['depth_route_1'], how='all'), x='smiles_nr_explicit_atoms')
    fig.update_layout(width=1000, height=600)
    fig.write_image(f"{output_folder}/nr_explicit_atoms.pdf")
    time.sleep(10)
    fig.write_image(f"{output_folder}/nr_explicit_atoms.pdf") 


    # Bin values
    # Binned cost (lowest_cost_route_found)
    nr_atoms_variable = 'smiles_nr_explicit_atoms'
    binned_var_name = 'smiles_nr_explicit_atoms_(binned)'
    num_bins = 10
    min_value = int(data_all_target[nr_atoms_variable].min()) 
    max_value = int(data_all_target[nr_atoms_variable].max())
    bin_range =np.linspace(min_value, max_value, num_bins+1, dtype=int)
    bin_labels = [f'{str(int(round(lower,0))).zfill(3)}-{str(int(round(upper,0))).zfill(3)}' for lower, upper in zip(bin_range[:-1], bin_range[1:])]

    data_all_target[binned_var_name] = pd.cut(data_all_target[nr_atoms_variable], 
                                            bins=bin_range, 
                                            labels=bin_labels, 
                                            include_lowest=True)


    # import random
    # random.set_seed(seed)
    eps = 0.1
    data_target_sorted_to_plot = data_all_target.copy()
    jitter = np.random.uniform(low=-eps/4, high=eps/4, size=len(data_target_sorted_to_plot.index))
    data_target_sorted_to_plot['dummy_ones'] = 1 + jitter

    fig = go.Figure()
    df1 = data_target_sorted_to_plot.dropna(subset=['depth_route_1'], how='all')
    df2 = data_target_sorted_to_plot.dropna(subset=['depth_route_2'], how='all')
    df3 = data_target_sorted_to_plot.dropna(subset=['depth_route_3'], how='all')
    fig.add_trace(go.Scatter(x=df1[rank_col], 
                            y=df1['dummy_ones']+eps,mode='markers', name='route_1'))
    fig.add_trace(go.Scatter(x=df2[rank_col], 
                            y=df2['dummy_ones'],mode='markers', name='route_2'))
    fig.add_trace(go.Scatter(x=df3[rank_col], 
                            y=df3['dummy_ones']-eps,mode='markers', name='route_3'))

    fig.update_traces(opacity=0.75)
    # fig.update_layout(barmode='group')
    fig.update_xaxes(type="log", title='Purchasable molecule rank according to distance from target. Ranges between 1 and dim_inventory (i.e. 13325). (log scale)')
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Rank of purchasable molecules in routes, according to distance to the target (log-x scale)", width=1000, height=600)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_2_3_log_scale.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_2_3_log_scale.pdf")
    # fig.show()

    nbinsx = 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_1'], how='all')[rank_col], 
                            nbinsx=nbinsx, name='route_1'))
    fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_2'], how='all')[rank_col], 
                            nbinsx=nbinsx, name='route_2'))
    fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_3'], how='all')[rank_col], 
                            nbinsx=nbinsx, name='route_3'))

    fig.update_traces(opacity=0.75)
    fig.update_layout(barmode='group', width=1000, height=600)
    # fig.update_xaxes(type="log", range=[0,np.log(max_rank)])
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_2_3.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_2_3.pdf") 
    # fig.show()


    nbinsx = 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_1'], how='all')[rank_col], 
                            nbinsx=nbinsx, name='route_1'))
    # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_2'], how='all')[rank_col], 
    #                            nbinsx=nbinsx, name='route_2'))
    # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_3'], how='all')[rank_col], 
    #                            nbinsx=nbinsx, name='route_3'))

    fig.update_traces(opacity=0.75)
    # fig.update_layout(barmode='group')
    # fig.update_xaxes(type="log", range=[0,np.log(max_rank)])
    fig.update_xaxes(title='Purchasable molecule rank according to distance from target. Ranges between 1 and dim_inventory (i.e. 13325).')
    # fig.update_yaxes(visible=False)
    fig.update_layout(title="Rank of purchasable molecules in routes, according to distance to the target")
    fig.update_layout(showlegend=True, width=1000, height=600)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1.pdf")
    # fig.show()

    # Split by depth
    # nbinsx = 100

    fig = px.histogram(data_all_target.dropna(subset=['depth_route_1'], how='all').sort_values('depth_route_1'), 
                    x=rank_col, color="depth_route_1",
                    marginal="box", # can be 'rug' `box`, `violin`
                            hover_data=data_all_target.columns)

    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_1'], how='all')['distance_to_target_rank'], 
    #                            nbinsx=nbinsx, name='route_1'))
    # # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_2'], how='all')['distance_to_target_rank'], 
    # #                            nbinsx=nbinsx, name='route_2'))
    # # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_3'], how='all')['distance_to_target_rank'], 
    # #                            nbinsx=nbinsx, name='route_3'))

    # fig.update_traces(opacity=0.75)
    fig.update_layout(
        dict(
            barmode='stack', 
            showlegend=True,
            title="Rank of purchasable molecules in routes, according to distance to the target - Split by depth",
            width=1000, height=600
        ))
    # fig.update_xaxes(type="log", range=[0,np.log(max_rank)])
    fig.update_xaxes(title='Purchasable molecule rank according to distance from target. Ranges between 1 and dim_inventory (i.e. 13325).',row=1, col=1)
    fig.update_yaxes(title='Number of molecules',row=1, col=1)
    # fig.update_yaxes(type="log")
    # fig.update_layout(title="Rank of purchasable molecules in routes, according to distance to the target")
    # fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(
        title="Depth in route1",
    ))
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_split_depth.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_split_depth.pdf") 
    # fig.show()


    # Split by complexity
    # nbinsx = 100

    fig = px.histogram(data_all_target.dropna(subset=['depth_route_1'], how='all').sort_values('smiles_nr_explicit_atoms_(binned)'), 
                    x=rank_col, color="smiles_nr_explicit_atoms_(binned)",
                    marginal="box", # can be 'rug' `box`, `violin`
                            hover_data=data_all_target.columns)

    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=data_all_target.dropna(subset=['depth_route_1'], how='all')['distance_to_target_rank'], 
    #                            nbinsx=nbinsx, name='route_1'))
    # # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_2'], how='all')['distance_to_target_rank'], 
    # #                            nbinsx=nbinsx, name='route_2'))
    # # fig.add_trace(go.Histogram(x=data_target_sorted_to_plot.dropna(subset=['depth_route_3'], how='all')['distance_to_target_rank'], 
    # #                            nbinsx=nbinsx, name='route_3'))

    # fig.update_traces(opacity=0.75)
    fig.update_layout(
        dict(
            barmode='stack', 
            showlegend=True,
            title="Rank of purchasable molecules in routes, according to distance to the target - Split by complexity",
            width=1000, height=600
        ))
    # fig.update_xaxes(type="log", range=[0,np.log(max_rank)])
    fig.update_xaxes(title='Purchasable molecule rank according to distance from target. Ranges between 1 and dim_inventory (i.e. 13325).',row=1, col=1)
    fig.update_yaxes(title='Number of molecules',row=1, col=1)
    # fig.update_yaxes(type="log")
    # fig.update_layout(title="Rank of purchasable molecules in routes, according to distance to the target")
    # fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(
        title="Nr explicit atoms"
        
    ))

    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_split_complex.pdf")
    time.sleep(10) 
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_1_split_complex.pdf")
    # fig.show()








