import math
import random
import plotly.graph_objects as go
import plotly.express as px
# import plotly.offline as pyo
# pyo.init_notebook_mode()
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

from create_plot_distances_histograms import concatenate_target_routes_data



if __name__ == "__main__":
    # distance_to_consider = 'Tanimoto'
    distance_to_consider = 'Fnps-nn'
    # distance_to_consider = 'Gnn'

    if distance_to_consider=='Tanimoto':
        distance_col = "Tanimoto_distance_to_target"
        rank_col = 'Tanimoto_distance_to_target_rank'
    elif distance_to_consider=='Fnps-nn':
        distance_col = "Fnps_distance_to_target"
        rank_col = 'Fnps_distance_to_target_rank'
    elif distance_to_consider=='Gnn':
        distance_col = "Gnn_distance_to_target"
        rank_col = 'Gnn_distance_to_target_rank'
        
        

    run_id = '202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7'
    # input_file = f'Runs/{run_id}/routes_df.csv'
    # routes_df = pd.read_csv(input_file)

    input_file_routes = f'Runs/{run_id}/targ_routes.pickle'
    input_file_distances = f'Runs/{run_id}/targ_to_purch_distances_v2.pickle'


    output_folder = f'Plots/{run_id}/InRoute_vs_not_v2/{distance_to_consider}'

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
    # Routes data
    with open(input_file_routes, 'rb') as handle:
        targ_routes_dict = pickle.load(handle)
        
    # Load distances data
    with open(input_file_distances, 'rb') as handle:
        distances_dict = pickle.load(handle)


    # Inventory
    from paroutes import PaRoutesInventory
    inventory=PaRoutesInventory(n=5)
    purch_smiles = [mol.smiles for mol in inventory.purchasable_mols()]
    len(purch_smiles)


    # Add rows non-purchasable molecules
    targ_routes_with_non_purch_dict = {}

    for target, target_routes_dict in targ_routes_dict.items():
        targ_routes_with_non_purch_dict[target] = {}
        for route_id, target_route_df in target_routes_dict.items():
    #         target_route_df = target_route_df.drop(columns=[distance_col, rank_col])
            
            # Add all purchasable molecules not in route
            purch_in_route = list(target_route_df['smiles'])
            purch_not_in_route = [purch_smile for purch_smile in purch_smiles if purch_smile not in purch_in_route]
            df_to_concat = pd.DataFrame({'smiles': purch_not_in_route, 'label': 'not_in_route', 'depth': np.nan})

            target_route_df = pd.concat([target_route_df, df_to_concat])
            target_route_df = pd.merge(target_route_df, distances_dict[target], how='left', on='smiles')
            target_route_df.loc[target_route_df['smiles']==target, distance_col] = 0
            target_route_df['target'] = target            

            targ_routes_with_non_purch_dict[target].update({route_id: target_route_df})
        

    # 1. Across all targets
    first = True
    sample_size = 500
    random.seed(1994)
    # targ_routes_with_non_purch_dict_sample = random.sample(targ_routes_with_non_purch_dict.items(),sample_size)
    items = list(targ_routes_with_non_purch_dict.items())
    sampled_items = random.sample(items, k=sample_size)
    targ_routes_with_non_purch_dict_sample = dict(sampled_items)
    
    for target, target_dict in tqdm(targ_routes_with_non_purch_dict_sample.items()):
        for route, target_route_df in target_dict.items():
            if first:
                targ_routes_with_non_purch_df = target_route_df
                first = False
            else: 
                targ_routes_with_non_purch_df = pd.concat([targ_routes_with_non_purch_df, target_route_df])
        

    # # Nice - commented out because RAM heavy
    # routes_to_consider = ['route_1', 'route_2', 'route_3']
    # col_to_plot = 'Tanimoto_distance_from_target'

    # data_no_target = targ_routes_with_non_purch_df.loc[targ_routes_with_non_purch_df['label']!='Target'].reset_index()

    # distances_in_route = data_no_target.loc[data_no_target['label'].isin(routes_to_consider), col_to_plot]
    # distances_not_in_route = data_no_target.loc[data_no_target['label']=='not_in_route', col_to_plot]
    
    # # Create the scatter plot
    # fig = go.Figure()

    # fig.add_trace(go.Box(
    #     y=distances_in_route,
    #     name='in_route',
    #     boxpoints='all',
    #     marker=dict(color='green'),
    #     line=dict(color='green')
    # ))

    # fig.add_trace(go.Box(
    #     y=distances_not_in_route,
    #     name='not_in_route',
    #     boxpoints='all',
    #     marker=dict(color='red'),
    #     line=dict(color='red')
    # ))


    # # Add the count of points on top of each box plot
    # fig.add_annotation(
    #     x=0,  # x-axis position of annotation
    #     y=1.1*max(distances_not_in_route),  # y-axis position of annotation
    #     text=f'In route count: {len(distances_in_route.index)}',  # annotation text
    #     showarrow=False,  # remove arrow
    #     font=dict(color='green')  # font color for 'in_route' count
    # )

    # fig.add_annotation(
    #     x=1,  # x-axis position of annotation
    #     y=1.1*max(distances_not_in_route),  # y-axis position of annotation
    #     text=f'Not in route count: {len(distances_not_in_route.index)}',  # annotation text
    #     showarrow=False,  # remove arrow
    #     font=dict(color='red')  # font color for 'not_in_route' count
    # )

    # # Display the plot
    # fig.show()



    # 1A - Only route 1 - All the negative points
    routes_to_consider = ['route_1']
    col_to_plot = rank_col

    data_no_target = targ_routes_with_non_purch_df.loc[targ_routes_with_non_purch_df['label']!='Target'].reset_index()

    distances_in_route = data_no_target.loc[data_no_target['label'].isin(routes_to_consider), col_to_plot]
    distances_not_in_route = data_no_target.loc[data_no_target['label']=='not_in_route', col_to_plot]


    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Violin(
        y=distances_in_route,
        name='in_route',
        box=dict(visible=True),
        points=False,
        spanmode = "hard",
        marker=dict(color=px.colors.qualitative.Plotly[9]),
        line=dict(color=px.colors.qualitative.Plotly[9])
    ))

    fig.add_trace(go.Violin(
        y=distances_not_in_route,
        name='not_in_route',
        box=dict(visible=True),
        points=False,
        spanmode = "hard",
        marker=dict(color=px.colors.qualitative.Plotly[0]),
        line=dict(color=px.colors.qualitative.Plotly[0])
    ))


    # Add the count of points on top of each box plot
    fig.add_annotation(
        x=0,  # x-axis position of annotation
        y=1.1*max(distances_not_in_route),  # y-axis position of annotation
        text=f'In route count: {len(distances_in_route.index)}',  # annotation text
        showarrow=False,  # remove arrow
        font=dict(color=px.colors.qualitative.Plotly[9])  # font color for 'in_route' count
    )

    fig.add_annotation(
        x=1,  # x-axis position of annotation
        y=1.1*max(distances_not_in_route),  # y-axis position of annotation
        text=f'Not in route count: {len(distances_not_in_route.index)}',  # annotation text
        showarrow=False,  # remove arrow
        font=dict(color=px.colors.qualitative.Plotly[0])  # font color for 'not_in_route' count
    )

    # Display the plot
    fig.update_layout(title="Rank of purchasable molecules in route vs not in route", width=1000, height=600)
    fig.update_yaxes(title='Purchasable molecule rank according to distance from target. </br></br> Ranges between 1 and dim_inventory (i.e. 13325)')
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_vs_not.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/{distance_to_consider}_rank_in_route_vs_not.pdf") 

    # fig.show()


        