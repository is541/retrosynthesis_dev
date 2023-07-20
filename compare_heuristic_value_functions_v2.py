"""
Somewhat enhanced value function comparison script.

Writes results to json output and does better caching.
However, this makes wallclock time unreliable because the cache is not really reset.
"""

from __future__ import annotations

import argparse
import gc
import logging
import json
import sys
from typing import Any
from tqdm import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.base import BaseNodeEvaluator
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.reaction_models.base import BackwardReactionModel

from paroutes import PaRoutesInventory, PaRoutesModel
from example_paroutes import PaRoutesRxnCost
from heuristic_value_functions import ScaledSAScoreCostFunction
from compare_heuristic_value_functions_v1 import (
    ScaledTanimotoNNAvgCostEstimator,
    FiniteMolIsPurchasableCost,
)
from faster_retro_star import ReduceValueFunctionCallsRetroStar
from neighbour_value_functions import DistanceToCost


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=25,
        help="Allowed number of calls to reaction model.",
    )
    parser.add_argument(
        "--paroutes_n",
        type=int,
        default=5,
        help="Which PaRoutes benchmark to use.",
    )
    parser.add_argument(
        "--and_node_cost_function",
        type=str,
        default="constant-1",
        help="Which cost function to use for AND nodes.",
    )
    return parser


def run_graph_retro_star(
    smiles_list: list[str],
    value_functions: list[tuple[str, BaseNodeEvaluator]],
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    rxn_cost_fn: BaseNodeEvaluator,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 100,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Runs graph retro star on a list of SMILES strings and reports the time of first solution."""

    logger = logging.getLogger("retro-star-run")

    high_integer = int(1e10)
    algs = [
        ReduceValueFunctionCallsRetroStar(
            reaction_model=rxn_model,
            mol_inventory=inventory,
            limit_reaction_model_calls=limit_rxn_model_calls,
            limit_iterations=high_integer,
            max_expansion_depth=high_integer,
            prevent_repeat_mol_in_trees=False,
            unique_nodes=True,
            and_node_cost_fn=rxn_cost_fn,
            value_function=vf,
            or_node_cost_fn=FiniteMolIsPurchasableCost(non_purchasable_cost=1e4),
            stop_on_first_solution=True,
        )
        for _, vf in value_functions
    ]
    if use_tqdm:
        smiles_iter = tqdm(
            smiles_list,
            dynamic_ncols=True,  # avoid issues open tmux on different screens
            smoothing=0.0,  # average speed, needed because searches vary a lot in length
        )
    else:
        smiles_iter = smiles_list

    output: dict[str, dict[str, dict[str, Any]]] = {
        name: dict() for name, _ in value_functions
    }
    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")

        # Potential reset reaction model.
        # However, do it only for new SMILES, since it is useful to share the cache between searches
        # on the same molecule
        _cache_size = len(rxn_model._cache)
        logger.debug(f"Reaction model cache size: {_cache_size}")
        if _cache_size > 1e6:
            logger.debug("Resetting reaction model cache.")
            rxn_model._cache.clear()
        for (name, _), alg in zip(value_functions, algs):
            # Do a pseudo-reset of the reaction model (keeping the actual cache intact)
            alg.reaction_model._num_cache_hits = 0
            alg.reaction_model._num_cache_misses = 0
            assert alg.reaction_model.num_calls() == 0

            # Do the search and record the time for the first solution
            output_graph, _ = alg.run_from_mol(Molecule(smiles))
            for node in output_graph.nodes():
                node.data["analysis_time"] = node.data["num_calls_rxn_model"]
                del node  # to not interfere with garbage collection below
            soln_time = get_first_solution_time(output_graph)
            output[name][smiles] = {
                "solution_time": soln_time,
                "num_nodes": len(output_graph),
            }
            logger.debug(
                f"Done {name+':':<30s} nodes={len(output_graph):>8d}, solution time = {soln_time:>8.3g}."
            )

            # Garbage collection
            del output_graph
            gc.collect()

    return output


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )
    logging.getLogger().info(args)

    # Load all SMILES to test
    with open(args.smiles_file, "r") as f:
        test_smiles = [line.strip() for line in f.readlines()]

    # Make reaction model, inventory, value functions
    rxn_model = PaRoutesModel()
    rxn_model.count_cache_in_num_calls = (
        True  # required to get correct num_calls. Only ok because it is a *graph* alg
    )
    inventory = PaRoutesInventory(n=args.paroutes_n)
    if args.and_node_cost_function == "constant-1":
        and_node_cost_fn = ConstantNodeEvaluator(1.0)
    elif args.and_node_cost_function == "paroutes":
        and_node_cost_fn = PaRoutesRxnCost()
    else:
        raise NotImplementedError(args.and_node_cost_function)

    # Create all the value functions to test
    value_fns = [  # baseline: 0 value function
        ("constant-0", ConstantNodeEvaluator(0.0)),
    ]

    # Nearest neighbour cost heuristics
    for num_nearest_neighbours in [1]:
        for scale in [1.0]:
            # Tanimoto distance cost heuristic
            value_fns.append(
                (
                    f"Tanimoto-top{num_nearest_neighbours}NN-linear-{scale}",
                    ScaledTanimotoNNAvgCostEstimator(
                        scale=scale,
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.NOTHING,
                        num_nearest_neighbours=num_nearest_neighbours,
                        nearest_neighbour_cache_size=100_000,
                    ),
                )
            )

    # SAscore cost heuristic (different scale)
    for scale in [1.0]:
        value_fns.append(
            (
                f"SAscore-linear-{scale}",
                ScaledSAScoreCostFunction(
                    scale=scale,
                ),
            )
        )

    # Run each value function
    overall_results = dict(
        args=args.__dict__,
    )
    overall_results["results"] = run_graph_retro_star(
        smiles_list=test_smiles,
        value_functions=value_fns,
        rxn_model=rxn_model,
        inventory=inventory,
        rxn_cost_fn=and_node_cost_fn,
        use_tqdm=True,
        limit_rxn_model_calls=args.limit_rxn_model_calls,
    )

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(overall_results, f, indent=2)
