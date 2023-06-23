"""
"""
from __future__ import annotations

import pickle
import os
from tqdm.auto import tqdm
from syntheseus.search.graph.message_passing import run_message_passing
from syntheseus.search.analysis.tree_solution_counting import num_solutions_update

# from syntheseus.search.analysis.route_extraction import min_cost_routes
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


if __name__ == "__main__":
    # Create df routes from search graph
    run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"
    input_folder = f"Runs/{run_id}/constant0_graph_pickles"
    output_file = f"Runs/{run_id}/routes_df.csv"

    num_top_routes_to_extract = 3

    cost_type = "cost_1_react"
    # cost_type = "cost_react_from_data"
    # cost_type = "cost_react_from_data_pow01"

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
                        # CHECK is there a more efficient implementation using message_passing instead of custom_assign_cost?
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
