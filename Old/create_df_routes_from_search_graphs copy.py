"""
"""
from __future__ import annotations

import pickle
import os
from tqdm.auto import tqdm
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.analysis.tree_solution_counting import num_solutions_update

from syntheseus.search.graph.and_or import OrNode, AndNode
from syntheseus.search.analysis.route_extraction import (
    _iter_top_routes,
    _min_route_cost,
    _min_route_partial_cost,
)
import numpy as np
import pandas as pd


def custom_assign_cost(graph, cost_type):
    if cost_type == "cost_1_react":
        for node in graph._graph.nodes():
            if isinstance(node, (AndNode,)):
                node.data["route_cost"] = 1.0
            else:
                node.data["route_cost"] = 0.0
    elif cost_type == "cost_react_from_data":
        for node in graph._graph.nodes():
            if isinstance(node, (AndNode,)):
                node.data["route_cost"] = node.data["retro_star_rxn_cost"]
            else:
                node.data["route_cost"] = 0.0
    elif cost_type == "cost_react_from_data_pow01":
        for node in graph._graph.nodes():
            if isinstance(node, (AndNode,)):
                node.data["route_cost"] = np.power(
                    node.data["retro_star_rxn_cost"], 0.1
                )
            else:
                node.data["route_cost"] = 0.0
    else:
        raise NotImplementedError(f"Cost type {cost_type}")

    return graph


def compute_cost_below_node(node, graph, route, total, cost_type):
    successors = graph.successors(node)

    if successors is None:
        return total

    for successor in successors:
        if successor in route:
            if cost_type == "cost_1_react":
                if isinstance(successor, (AndNode,)):
                    total += 1
                else:
                    total += 0
            elif cost_type == "cost_react_from_data":
                if isinstance(successor, (AndNode,)):
                    total += successor.data["retro_star_rxn_cost"]
                else:
                    total += 0
            elif cost_type == "cost_react_from_data_pow01":
                if isinstance(successor, (AndNode,)):
                    total += np.power(successor.data["retro_star_rxn_cost"], 0.1)
                else:
                    total += 0
            else:
                raise NotImplementedError(f"Cost type {cost_type}")
            total = compute_cost_below_node(successor, graph, route, total, cost_type)

    return total

def create_df_routes(input_folder, num_top_routes_to_extract, cost_type, output_file):
    with open(output_file, "w") as f:
        f.write(
            "target_smiles,n_routes_found,route_rank,route_cost,"
            "intermediate_smiles,intermediate_is_purchasable,"
            "intermediate_parent,intermediate_depth,intermediate_cost_below\n"
        )
        smiles_id = 0
        for file_name in tqdm(
            [file for file in os.listdir(input_folder) if "pickle" in file]
        ):
            with open(f"{input_folder}/{file_name}", "rb") as handle:
                for smiles, output_graph in (pickle.load(handle)).items():
                    # assert smiles==output_graph.root_node.mol.smiles, f"smiles: {smiles} is different from root node smiles: {output_graph.root_node.mol.smiles}"

                    # Count routes
                    run_message_passing(
                        graph=output_graph,
                        nodes=sorted(
                            output_graph.nodes(),
                            key=lambda node: node.depth,
                            reverse=True,
                        ),
                        update_fns=[
                            num_solutions_update,
                        ],  # type: ignore[list-item]  # confusion about AndOrGraph type
                        update_predecessors=True,
                        update_successors=False,
                    )
                    n_routes_found = output_graph.root_node.data["num_routes"]
                    assert (
                        output_graph.root_node.has_solution & n_routes_found > 0
                    ) | (
                        (not output_graph.root_node.has_solution) & n_routes_found == 0
                    )
                    if n_routes_found > 0:
                        # Assign costs and extract top routes
                        output_graph = custom_assign_cost(output_graph, cost_type)

                        for route_rank, (route_cost, route) in enumerate(
                            _iter_top_routes(
                                graph=output_graph,
                                cost_fn=_min_route_cost,
                                cost_lower_bound=_min_route_partial_cost,
                                max_routes=num_top_routes_to_extract,
                                yield_partial_routes=False,
                            )
                        ):
                            #                         # Visualise route
                            #                         visualization.visualize_andor(
                            #                             graph=output_graph,
                            #                             filename=f"graphs/smiles_{smiles_id}_{route_rank+1}.pdf",
                            #                             nodes=route,
                            #                         )
                            # Iterate trough the route
                            for node in route:
                                if isinstance(node, OrNode):
                                    intermediate_smiles = node.mol.smiles
                                    intermediate_is_purchasable = node.mol.metadata.get(
                                        "is_purchasable"
                                    )
                                    intermediate_depth = node.depth

                                    if intermediate_depth > 0:
                                        intermediate_parent = next(
                                            output_graph.predecessors(
                                                next(output_graph.predecessors(node))
                                            )
                                        ).mol.smiles
                                    else:
                                        intermediate_parent = ""
                                    intermediate_cost_below = compute_cost_below_node(
                                        node=node,
                                        graph=output_graph,
                                        route=route,
                                        total=0,
                                        cost_type="cost_1_react",
                                    )

                                    output_str = (
                                        f"{smiles},{n_routes_found},{route_rank+1},{route_cost},"
                                        f"{intermediate_smiles},{intermediate_is_purchasable},"
                                        f"{intermediate_parent},{intermediate_depth},{intermediate_cost_below}\n"
                                    )
                                    f.write(output_str)
                                    f.flush()
                        else:
                            output_str = f"{smiles},{n_routes_found},,," f",," f",,\n"
                            f.write(output_str)
                            f.flush()

                        smiles_id += 1

# Create dict with sets of purchasable molecules in each of the top 3 routes (for every target)
def create_routes_dict(routes_df, target_smiles, cols_to_keep_renamed, output_file_routes):
    routes_data_dict = {}
    for target in tqdm(target_smiles):
        target_df = routes_df.loc[routes_df["target_smiles"] == target]
        routes_data_dict[target] = {}

        for route_rank in target_df["route_rank"].dropna().unique():
            target_route_df = target_df.loc[
                (
                    (target_df["intermediate_is_purchasable"])
                    | (target_df["intermediate_smiles"] == target)
                )
                & (target_df["route_rank"] == route_rank),
                cols_to_keep_renamed.keys(),
            ].drop_duplicates()
            route_name = "route_" + str(int(route_rank))
            target_route_df["route_rank"] = route_name
            #         target_route_df['route_rank'] = 'route_' + target_route_df['route_rank'].astype(int).astype(str)

            target_route_df = target_route_df.rename(columns=cols_to_keep_renamed)

            # target_route_df = pd.merge(
            #     target_route_df, distance_df_sorted, how="left", on="smiles"
            # )

            target_mask = target_route_df["smiles"] == target
            target_route_df.loc[target_mask, "label"] = "Target"
            # target_route_df.loc[target_mask, "Tanimoto_distance_from_target"] = 0

            routes_data_dict[target].update({route_name: target_route_df})

        with open(output_file_routes, "wb") as handle:
            pickle.dump(routes_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    # Create df routes from search graph
    # dataset_str = 'paroutes'
    dataset_str = 'guacamol'
    # PAROUTES or GUACAMOL
    if dataset_str=='paroutes':
        run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"
    elif dataset_str=='guacamol':
        run_id = "Guacamol_combined"
    input_folder = f"Runs/{run_id}/constant0_graph_pickles"
    
    
    # 1. Create df_routes
    output_file = f"Runs/{run_id}/routes_df.csv"

    num_top_routes_to_extract = 3

    cost_type = "cost_1_react"
    # cost_type = "cost_react_from_data"
    # cost_type = "cost_react_from_data_pow01"
    
    create_df_routes(input_folder, num_top_routes_to_extract, cost_type, output_file)
    
    # 2. create routes dict (reading the file just created)
    input_file = f"Runs/{run_id}/routes_df.csv"
    routes_df = pd.read_csv(input_file)
    output_file_routes = f"Runs/{run_id}/targ_routes.pickle"
    
    target_smiles = routes_df["target_smiles"].unique()
    cols_to_keep_renamed = {
        "route_rank": "label",
        "intermediate_smiles": "smiles",
        "intermediate_depth": "depth",
    }

    create_routes_dict(routes_df, target_smiles, cols_to_keep_renamed, output_file_routes)

    
