"""
Script to find minimum cost routes for PaRoutes benchmark
on a list of SMILES provided by the user.
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime
from uuid import uuid4
import pickle
import numpy as np

from tqdm.auto import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.node_evaluation.base import (
    # BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)

from paroutes import PaRoutesInventory, PaRoutesModel
from faster_retro_star import ReduceValueFunctionCallsRetroStar
# from example_paroutes import PaRoutesRxnCost

class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Text file with SMILES to run search on."
        " One SMILES per line, no header.",
    )
    # parser.add_argument(
    #     "--output_csv_path", type=str, required=True, help="CSV file to output to."
    # )
    parser.add_argument(
        "--limit_iterations",
        type=int,
        default=10_000,
        help="Maximum number of algorithm iterations.",
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=1000,
        help="Allowed number of calls to reaction model.",
    )
    parser.add_argument(
        "--paroutes_n",
        type=int,
        default=5,
        help="Which PaRoutes benchmark to use.",
    )
    parser.add_argument(
        "--limit_num_smiles",
        type=int,
        default=None,
        help="Maximum number of SMILES to run.",
    )
    parser.add_argument(
        "--start_from_smiles_index_n",
        type=int,
        default=None,
        help="Consider only target SMILES after index n (n=0 equivalent to consider all SMILES).",
    )
    parser.add_argument(
        "--save_output_pickle",
        type=bool,
        default=True,
        help="Whether to save the pickle of the output graphs of each target molecule",
    )
    args = parser.parse_args()

    # Create eventid
    eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    output_folder = f"Runs/{eventid}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if args.save_output_pickle:
            os.makedirs(f"{output_folder}/constant0_graph_pickles")

    # Logging
    # logging.basicConfig(
    #     stream=sys.stdout,
    #     level=logging.INFO,
    #     format="%(asctime)s %(name)s %(levelname)s %(message)s",
    #     filemode="w",
    # )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f"{output_folder}/logs.txt", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger.info(args)

    # Read input SMILES
    with open(args.input_file, "r") as f:
        test_smiles = [line.strip() for line in f.readlines()]

    if args.limit_num_smiles is not None:
        test_smiles = test_smiles[: args.limit_num_smiles]

    # Make reaction model, inventory, value functions, algorithm
    rxn_model = PaRoutesModel()
    inventory = PaRoutesInventory(n=args.paroutes_n)
    value_function = ConstantNodeEvaluator(0.0)
    algorithm = ReduceValueFunctionCallsRetroStar(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=args.limit_rxn_model_calls,
        limit_iterations=args.limit_iterations,
        max_expansion_depth=15,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=True,  # original paper did this
        and_node_cost_fn=PaRoutesRxnCost(),
        value_function=value_function,
        terminate_once_min_cost_route_found=True,
    )

    # Run search on all SMILES
    output_path_csv = f"{output_folder}/{eventid}_results.csv"
    with open(output_path_csv, "w") as f:
        f.write(
            "smiles,n_iter,first_soln_time,lowest_cost_route_found,"
            "best_route_cost_lower_bound,lowest_depth_route_found,"
            "best_route_depth_lower_bound,num_calls_rxn_model,num_nodes_in_tree\n"
        )
        if args.start_from_smiles_index_n is not None:
            index_start = args.start_from_smiles_index_n
        else:
            index_start = 0
        for i, smiles in enumerate(test_smiles[index_start:], start=index_start):
            print("Smiles index: ", i)
            print(smiles)
            # Run search
            algorithm.reset()
            output_graph, n_iter = algorithm.run_from_mol(Molecule(smiles))

            # Save output graph as pickle
            if args.save_output_pickle:
                with open(
                    f"{output_folder}/constant0_graph_pickles/mol_{i}.pickle", "wb"
                ) as handle:
                    pickle.dump({smiles: output_graph}, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Solution time
            for node in output_graph.nodes():
                node.data["analysis_time"] = node.data["num_calls_rxn_model"]
            first_soln_time = get_first_solution_time(output_graph)

            # Min cost routes
            lowest_cost_route_found = output_graph.root_node.data[
                "retro_star_proven_reaction_number"
            ]
            best_route_cost_lower_bound = output_graph.root_node.data["reaction_number"]

            # Min cost in terms of depth.
            # First, change the reaction costs and run retro star updates again
            for node in output_graph.nodes():
                if isinstance(node, AndNode):
                    node.data["retro_star_rxn_cost"] = 1.0
            algorithm.set_node_values(output_graph.nodes(), output_graph)

            # Second, read off min costs in terms of depth
            lowest_depth_route_found = output_graph.root_node.data[
                "retro_star_proven_reaction_number"
            ]
            best_route_depth_lower_bound = output_graph.root_node.data[
                "reaction_number"
            ]

            # Output everything
            output_str = (
                f"{smiles},{n_iter},{first_soln_time},"
                f"{lowest_cost_route_found},{best_route_cost_lower_bound},"
                f"{lowest_depth_route_found},{best_route_depth_lower_bound},"
                f"{algorithm.reaction_model.num_calls()},{len(output_graph)}\n"
            )
            f.write(output_str)
            f.flush()
            
            del output_graph
