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


# def custom_assign_cost(graph, cost_type):
#     if cost_type == "cost_1_react":
#         for node in graph._graph.nodes():
#             if isinstance(node, (AndNode,)):
#                 node.data["route_cost"] = 1.0
#             else:
#                 node.data["route_cost"] = 0.0
#     elif cost_type == "cost_react_from_data":
#         for node in graph._graph.nodes():
#             if isinstance(node, (AndNode,)):
#                 node.data["route_cost"] = node.data["retro_star_rxn_cost"]
#             else:
#                 node.data["route_cost"] = 0.0
#     elif cost_type == "cost_react_from_data_pow01":
#         for node in graph._graph.nodes():
#             if isinstance(node, (AndNode,)):
#                 node.data["route_cost"] = np.power(
#                     node.data["retro_star_rxn_cost"], 0.1
#                 )
#             else:
#                 node.data["route_cost"] = 0.0
#     else:
#         raise NotImplementedError(f"Cost type {cost_type}")

#     return graph


def create_dict_pos_neg_first_reaction(input_folder, keep_only_not_purchasable, output_file):
    result_dict = {}
    for file_name in tqdm([file for file in os.listdir(input_folder) if "pickle" in file]):
        with open(f"{input_folder}/{file_name}", "rb") as handle:
            for smiles, output_graph in (pickle.load(handle)).items():
                if output_graph.root_node.has_solution:
                    result_dict[smiles] = {}
                    result_dict[smiles]['positive_samples'] = []
                    result_dict[smiles]['negative_samples'] = []
                    # Extract best route
                    (best_route_cost, best_route) = next(_iter_top_routes(
                                    graph=output_graph,
                                    cost_fn=_min_route_cost,
                                    cost_lower_bound=_min_route_partial_cost,
                                    max_routes=1,
                                    yield_partial_routes=False,
                                ))
                    
                    # Positive is: first reaction of the best route and children molecules
                    positive_children = []
                    for node in best_route:
                        if isinstance(node, AndNode) & (node.depth==1):
                            best_reaction = node.reaction#.metadata["template_idx"]
                            best_reaction_cost = node.data["retro_star_rxn_cost"]
                        elif isinstance(node, OrNode):
                            if keep_only_not_purchasable:
                                check_purch = node.mol.metadata["is_purchasable"]
                            else:
                                check_purch = False                      
                            if (node.depth==2) & (not check_purch):
                                positive_children.append(node.mol.smiles)
                    result_dict[smiles]['positive_samples'].append(positive_children)
                    result_dict[smiles]['cost_pos_react'] = best_reaction_cost
                    
                    # Negatives are: all first reactions not chosen, along with the respective children
                    other_reaction_costs = []
                    for node in output_graph.successors(output_graph._root_node):
                        # if isinstance(node, AndNode) & (node.reaction.metadata["template_idx"]!=best_reaction_idx):
                        if isinstance(node, AndNode) & (node.reaction!=best_reaction):
                            reaction_children_nodes = output_graph.successors(node)
                            
                            if keep_only_not_purchasable:
                                reaction_children = [node.mol.smiles for node in reaction_children_nodes if not node.mol.metadata["is_purchasable"]]  
                            else:
                                reaction_children = [node.mol.smiles for node in reaction_children_nodes]
                            result_dict[smiles]['negative_samples'].append(reaction_children)
                            other_reaction_costs.append(node.data["retro_star_rxn_cost"])
                    result_dict[smiles]['cost_neg_react'] = other_reaction_costs
                else:
                    pass
    
    with open(output_file, "wb") as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    # Create df routes from search graph
    dataset_str = 'paroutes'
    # dataset_str = 'guacamol'
    # PAROUTES or GUACAMOL
    if dataset_str=='paroutes':
        run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"
    elif dataset_str=='guacamol':
        run_id = "Guacamol_combined"
    input_folder = f"Runs/{run_id}/constant0_graph_pickles"
    
    keep_only_not_purchasable = False
    if keep_only_not_purchasable:
        only_purch = "not_purch"
    else:
        only_purch = "all"
    
    output_dict = f"Runs/{run_id}/first_reaction_positive_negatives_" + only_purch + ".pickle"

    

    # cost_type = "cost_1_react"
    # # cost_type = "cost_react_from_data"
    # # cost_type = "cost_react_from_data_pow01"
    
    create_dict_pos_neg_first_reaction(input_folder, keep_only_not_purchasable, output_dict)
    
    
    
