"""
Demo script comparing nearest neighbour cost function with constant value function on PaRoutes.
"""
from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from tqdm.auto import tqdm

from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.analysis.route_extraction import min_cost_routes
from syntheseus.search.reaction_models.base import BackwardReactionModel
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.node_evaluation.base import (
    BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles
from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost
from faster_retro_star import ReduceValueFunctionCallsRetroStar
from analysis import calculate_min_synthesis_cost


class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()


def compare_cost_functions(
    smiles_list: list[str],
    value_functions: list[tuple[str, BaseNodeEvaluator]],
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 100,
    limit_iterations: int = 1_000_000,
) -> list[tuple[float, ...]]:
    """
    Do search on a list of SMILES strings and report the time of first solution.
    """

    # Initialize algorithm.
    common_kwargs = dict(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=limit_rxn_model_calls,
        limit_iterations=limit_iterations,
        max_expansion_depth=15,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=True,  # original paper did this
    )
    algs = [
        ReduceValueFunctionCallsRetroStar(
            and_node_cost_fn=PaRoutesRxnCost(), value_function=fn, **common_kwargs
        )
        for _, fn in value_functions
    ]

    # Do search
    logger = logging.getLogger("COMPARISON")
    min_soln_times: list[tuple[float, ...]] = []
    if use_tqdm:
        smiles_iter = tqdm(smiles_list)
    else:
        smiles_iter = smiles_list
    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")
        this_soln_times = list()
        for (name, _), alg in zip(value_functions, algs):
            alg.reset()
            output_graph, _ = alg.run_from_mol(Molecule(smiles))

            # Analyze solution time
            for node in output_graph.nodes():
                node.data["analysis_time"] = node.data["num_calls_rxn_model"]
            soln_time = get_first_solution_time(output_graph)
            this_soln_times.append(soln_time)

            # Analyze number of routes
            MAX_ROUTES = 10000
            routes = list(min_cost_routes(output_graph, MAX_ROUTES))

            # Analyze minimum synthesis cost
            # NOTE: changes reaction numbers in-place
            calculate_min_synthesis_cost(output_graph)

            if alg.reaction_model.num_calls() < limit_rxn_model_calls:
                note = " (NOTE: this was less than the maximum budget)"
            else:
                note = ""
            logger.debug(
                f"Done {name}: nodes={len(output_graph)}, solution time = {soln_time}, "
                f"min synthesis cost = {output_graph.root_node.data['reaction_number']}, "
                f"num routes = {len(routes)} (capped at {MAX_ROUTES}), "
                f"final num rxn model calls = {alg.reaction_model.num_calls()}{note}, "
                f"final num value model calls = {alg.value_function.num_calls}."
            )
        min_soln_times.append(tuple(this_soln_times))

    return min_soln_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit_num_smiles",
        type=int,
        default=None,
        help="Maximum number of SMILES to run.",
    )
    parser.add_argument(
        "--limit_iterations",
        type=int,
        default=500,
        help="Maximum number of algorithm iterations.",
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
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        filemode="w",
    )
    logging.getLogger().info(args)

    # Load all SMILES to test
    test_smiles = get_target_smiles(args.paroutes_n)
    if args.limit_num_smiles is not None:
        test_smiles = test_smiles[: args.limit_num_smiles]

    # Make reaction model, inventory, value functions
    rxn_model = PaRoutesModel()
    inventory = PaRoutesInventory(n=args.paroutes_n)
    value_fns = [
        ("constant-0", ConstantNodeEvaluator(0.0)),
        (
            "Tanimoto-distance",
            TanimotoNNCostEstimator(
                inventory=inventory, distance_to_cost=DistanceToCost.NOTHING
            ),
        ),
    ]

    # Run without value function (retro*-0)
    compare_cost_functions(
        smiles_list=test_smiles,
        value_functions=value_fns,
        limit_rxn_model_calls=args.limit_rxn_model_calls,
        limit_iterations=args.limit_iterations,
        use_tqdm=True,
        rxn_model=rxn_model,
        inventory=inventory,
    )
