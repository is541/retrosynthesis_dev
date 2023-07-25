import pandas as pd
import os

import plotly.express as px
import time

from value_functions import labelalias

def plot_correlation(df_results_tot, col_name, col_label):
    fig = px.violin(df_results_tot, x="lowest_depth_route_found", y=col_name, width=1000, height=600,
    #              points='all',
                    box=True,
                    color='is_solved',
                    color_discrete_sequence = [px.colors.qualitative.Plotly[1], px.colors.qualitative.Plotly[2]],
                    violinmode='overlay',
                labels={
    #                      "algorithm": None,
    #                      "value": "Time to first solution",
    #                      "species": "Species of Iris"
                },
    #              title="Time to first solution"
                )
    fig.update_layout(xaxis_title="Cost of minimal-cost route to syntesise target </br></br> (assignig cost=1 to each reaction and cost=0 to each molecule in route)", 
                    yaxis_title=f"{col_label} distance between taget </br></br> and closest purchasable molecule",
                    title="Correlation between nearest neighbour distance and minimal cost route"
                    )
    # fig.update_xaxes(labelalias=labelalias, categoryorder='array', categoryarray=list(labelalias.keys()))
    # fig.write_image(f'{output_folder}/Boxplot_time_first_solution.png') 
    fig.write_image(f"{output_folder}/cost_value_correlation_{col_label}.pdf") 
    time.sleep(10)
    fig.write_image(f"{output_folder}/cost_value_correlation_{col_label}.pdf") 
    # fig.show()

if __name__ == "__main__":
    
    eventid= '202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7'
    input_folder = f"Runs/{eventid}"

    output_folder = f'Plots/{eventid}/Cost_value_correlation_v2'

    if not os.path.exists(output_folder):
            os.makedirs(output_folder)


    # df_results_tot = pd.read_csv(f'{input_folder}/paroutes_n5_result_added_value_fns.csv')
    df_results_tot = pd.read_csv(f'{input_folder}/paroutes_result_added_value_fns_v2.csv')
    df_results_tot['is_solved'] = (df_results_tot['first_soln_time']!=-1 )

    values_to_consider = {
        "Tanimoto-distance": "Tanimoto",
        "Embedding-from-fingerprints": "Fnps-nn",
        "Embedding-from-gnn": "Gnn",
        "Retro*": "RetroStar"
    }

    for col_name, col_label in values_to_consider.items():
        plot_correlation(df_results_tot=df_results_tot, col_name=col_name, col_label=col_label)

