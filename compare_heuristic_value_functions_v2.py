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
import os
import torch
import deepchem as dc
import numpy as np
import pandas as pd


from syntheseus.search.chem import Molecule
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.base import BaseNodeEvaluator
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.reaction_models.base import BackwardReactionModel
from syntheseus.search.graph.and_or import AndOrGraph, OrNode, AndNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel
# from example_paroutes import PaRoutesRxnCost
# from heuristic_value_functions import ScaledSAScoreCostFunction
# from compare_heuristic_value_functions_v1 import (
#     # ScaledTanimotoNNAvgCostEstimator,
#     FiniteMolIsPurchasableCost,
# )
from faster_retro_star import ReduceValueFunctionCallsRetroStar, RetroStarSearch
from value_functions import initialize_value_functions, ConstantMolEvaluator
from embedding_model import (
    FingerprintModel,
    GNNModel,
    load_embedding_model_from_pickle,
    load_embedding_model_from_checkpoint,
)

class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()

class FiniteMolIsPurchasableCost(NoCacheNodeEvaluator[OrNode]):
    def __init__(self, non_purchasable_cost: float, **kwargs):
        super().__init__(**kwargs)
        self.non_purchasable_cost = non_purchasable_cost

    def _evaluate_nodes(  # type: ignore[override]
        self,
        nodes: Sequence[OrNode],
        graph: Optional[AndOrGraph] = None,
    ) -> list[float]:
        return [
            0.0
            if node.mol.metadata.get("is_purchasable")
            else self.non_purchasable_cost
            for node in nodes
        ]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
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
        default=5000,
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
        # default="constant-1",
        required=True,
        help="Which cost function to use for AND nodes.",
    )
    parser.add_argument(
        "--reduce_value_function_calls",
        # action="store_true",
        type=bool,
        default=False,
        help="Flag to use reduced value function retro star.",
    )
    parser.add_argument(
        "--fnp_embedding_model_to_use",
        type=str,
        # required=True,
        default="fnp_nn_0709_sampleInLoss",  # "fingerprints_v1" # "fnp_nn_0629"
    )
    parser.add_argument(
        "--gnn_embedding_model_to_use",
        type=str,
        # required=True,
        default="gnn_0709_sampleInLoss",
    )
    return parser


def create_df_from_json(data):
    algorithm = []
    similes = []
    properties = []
    values = []

    # Extract data from the JSON and store it in lists
    for alg, result in data["results"].items():
        for sim, props in result.items():
            for prop, val in props.items():
                algorithm.append(alg)
                similes.append(sim)
                properties.append(prop)
                values.append(val)

    # Create DataFrame
    df = pd.DataFrame({
        "algorithm": algorithm,
        "similes": similes,
        "property": properties,
        "value": values
    })

    # Filter and rename the 'solution_time' column to 'sol_time'
    df["property"] = df["property"].map({'solution_time': "sol_time"})
    df = df[df["property"] == "sol_time"]#.rename(columns={"value": "sol_time"}).drop(columns=["property"])
    
    return df

# def create_quantiles_from_df():⁄


def run_graph_retro_star(
    smiles_list: list[str],
    value_functions: list[tuple[str, BaseNodeEvaluator]],
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    rxn_cost_fn: BaseNodeEvaluator,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 100,
    reduce_value_function_calls: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Runs graph retro star on a list of SMILES strings and reports the time of first solution."""

    logger = logging.getLogger("retro-star-run")

    high_integer = int(1e10)
    if reduce_value_function_calls:
        alg_cls = ReduceValueFunctionCallsRetroStar
    else:
        alg_cls = RetroStarSearch
    algs = [
        alg_cls(
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
                "sol_time": soln_time,
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
    
    # Create output folder
    if args.input_file=="Guacamol_100_hardest_to_solve":
        eventid = f'202306-3017-5132-d4cd54bf-e5a4-44c5-82af-c629a3692d87_HARDEST_v2_cost_{args.and_node_cost_function}'
    if args.input_file=="Guacamol_100_hard_to_solve":
        eventid = f"202307-0320-4900-d4c27728-a5aa-4177-8681-268a17c3d208_HARD_v2_cost_{args.and_node_cost_function}"
    elif args.input_file=="Guacamol_100_mid_hard_to_solve":
        eventid = f'202307-0620-3725-cc7b1f07-14cd-47e8-9d40-f5b2f358fa28_MID_HARD_v2_cost_{args.and_node_cost_function}'
    elif args.input_file=="Guacamol_100_mid_to_solve":
        eventid = f'MID_v2_cost_{args.and_node_cost_function}'
    elif args.input_file=="Guacamol_100_mid_easy_to_solve":
        eventid = f'MID_EASY_v2_cost_{args.and_node_cost_function}'
    output_folder = f"CompareTanimotoLearnt/{eventid}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load all SMILES to test
    with open(f"Data/{args.input_file}.txt", "r") as f:
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

    
    value_fns_names = [
        'constant-0',
        'Tanimoto-distance',
        'Tanimoto-distance-TIMES0',
        'Tanimoto-distance-TIMES001',
        'Tanimoto-distance-TIMES01',
        'Tanimoto-distance-TIMES03',
        'Tanimoto-distance-TIMES10',
        'Tanimoto-distance-TIMES100',
        'Tanimoto-distance-TIMES1000',
        # 'Tanimoto-distance-EXP',
        # 'Tanimoto-distance-SQRT',
        # 'Tanimoto-distance-NUM_NEIGHBORS_TO_1',
        # "Embedding-from-fingerprints-TIMES01",
        # "Embedding-from-fingerprints-TIMES03",
        "Embedding-from-fingerprints",
        "Embedding-from-fingerprints-TIMES10",
        "Embedding-from-fingerprints-TIMES100",
        # "Embedding-from-fingerprints-TIMES1000",
        # "Embedding-from-fingerprints-TIMES10000",
        # "Embedding-from-gnn-TIMES01",
        # "Embedding-from-gnn-TIMES03",
        "Embedding-from-gnn",
        "Embedding-from-gnn-TIMES10",
        "Embedding-from-gnn-TIMES100",
        # "Embedding-from-gnn-TIMES1000",
        # "Embedding-from-gnn-TIMES10000",
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fingerprint model

    model_fnps, config_fnps = load_embedding_model_from_checkpoint(
        experiment_name=args.fnp_embedding_model_to_use,
        device=device,
        checkpoint_name="epoch_11_checkpoint",
    )

    # Graph model
    model_gnn, config_gnn = load_embedding_model_from_checkpoint(
        experiment_name=args.gnn_embedding_model_to_use,
        device=device,
        checkpoint_name="epoch_76_checkpoint",
    )

    distance_type_fnps = "cosine"
    distance_type_gnn = "cosine"
    featurizer_gnn = dc.feat.MolGraphConvFeaturizer()

    value_fns = initialize_value_functions(
        value_fns_names=value_fns_names,
        inventory=inventory,
        model_fnps=model_fnps,
        distance_type_fnps=distance_type_fnps,
        model_gnn=model_gnn,
        distance_type_gnn=distance_type_gnn,
        featurizer_gnn=featurizer_gnn,
        device=device,
    )

    # value_fns = [  # baseline: 0 value function
    #     ("constant-0", ConstantNodeEvaluator(0.0)),
    # ]
    # # Nearest neighbour cost heuristics
    # for num_nearest_neighbours in [1]:
    #     for scale in [1.0]:
    #         # Tanimoto distance cost heuristic
    #         value_fns.append(
    #             (
    #                 f"Tanimoto-top{num_nearest_neighbours}NN-linear-{scale}",
    #                 ScaledTanimotoNNAvgCostEstimator(
    #                     scale=scale,
    #                     inventory=inventory,
    #                     distance_to_cost=DistanceToCost.NOTHING,
    #                     num_nearest_neighbours=num_nearest_neighbours,
    #                     nearest_neighbour_cache_size=100_000,
    #                 ),
    #             )
    #         )
    

    # # SAscore cost heuristic (different scale)
    # for scale in [1.0]:
    #     value_fns.append(
    #         (
    #             f"SAscore-linear-{scale}",
    #             ScaledSAScoreCostFunction(
    #                 scale=scale,
    #             ),
    #         )
    #     )

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
    with open(f"{output_folder}/{args.output_json}", "w") as f:
        json.dump(overall_results, f, indent=2)
    
    # # Load data
    # with open(f"{output_folder}/{args.output_json}", 'r') as file:
    #     overall_results = json.load(file)  
    
    overall_results_df = create_df_from_json(overall_results)
    overall_results_df.to_csv(f"{output_folder}/results_all.csv", index=False)
