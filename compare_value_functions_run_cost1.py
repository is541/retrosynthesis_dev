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
import torch

from tqdm.auto import tqdm

# from deepchem.utils.save import log as dc_log


from syntheseus.search.chem import Molecule
from syntheseus.search.graph.and_or import AndNode
from syntheseus.search.analysis.solution_time import get_first_solution_time
from syntheseus.search.node_evaluation.base import (
    # BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)
import deepchem as dc

from paroutes import PaRoutesInventory, PaRoutesModel
from faster_retro_star import ReduceValueFunctionCallsRetroStar

from tqdm.auto import tqdm

from syntheseus.search.algorithms.best_first.retro_star import (
    RetroStarSearch,
    MolIsPurchasableCost,
)
from syntheseus.search.analysis.route_extraction import iter_routes_cost_order
from syntheseus.search.reaction_models.base import BackwardReactionModel
from syntheseus.search.mol_inventory import BaseMolInventory
from syntheseus.search.node_evaluation.base import (
    BaseNodeEvaluator,
    NoCacheNodeEvaluator,
)
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator

from paroutes import PaRoutesInventory, PaRoutesModel

# from neighbour_value_functions import TanimotoNNCostEstimator, DistanceToCost

from syntheseus.search.analysis import diversity
from value_functions import initialize_value_functions, ConstantMolEvaluator
from embedding_model import (
    FingerprintModel,
    GNNModel,
    load_embedding_model_from_pickle,
    load_embedding_model_from_checkpoint,
)

# from syntheseus.search.algorithms.best_first.retro_star import MolIsPurchasableCost


class SearchResult:
    def __init__(
        self,
        name,
        soln_time_dict,
        # num_different_routes_dict,
        # final_num_rxn_model_calls_dict,
        # final_num_value_function_calls_dict,
        # output_graph_dict,
        # routes_dict,
    ):
        self.name = name
        self.soln_time_dict = soln_time_dict
        # self.num_different_routes_dict = num_different_routes_dict
        # self.final_num_rxn_model_calls_dict = final_num_rxn_model_calls_dict
        # self.output_graph_dict = output_graph_dict
        # self.routes_dict = routes_dict
        # self.final_num_value_function_calls_dict = final_num_value_function_calls_dict


class PaRoutesRxnCost(NoCacheNodeEvaluator[AndNode]):
    """Cost of reaction is negative log softmax, floored at -3."""

    def _evaluate_nodes(self, nodes: list[AndNode], graph=None) -> list[float]:
        softmaxes = np.asarray([node.reaction.metadata["softmax"] for node in nodes])
        costs = np.clip(-np.log(softmaxes), 1e-1, 10.0)
        return costs.tolist()


def run_algorithm(
    name: str,
    smiles_list: list[str],
    value_function: BaseNodeEvaluator,
    rxn_model: BackwardReactionModel,
    inventory: BaseMolInventory,
    and_node_cost_fn: BaseNodeEvaluator[AndNode],
    or_node_cost_fn: BaseNodeEvaluator[OrNode],
    max_expansion_depth: int = 15,
    prevent_repeat_mol_in_trees: bool = True,
    use_tqdm: bool = False,
    limit_rxn_model_calls: int = 100,
    limit_iterations: int = 1_000_000,
    logger: logging.RootLogger = logging.getLogger(),
    stop_on_first_solution: bool = False,
) -> SearchResult:
    """
    Do search on a list of SMILES strings and report the time of first solution.
    """

    # Initialize algorithm.
    common_kwargs = dict(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=limit_rxn_model_calls,
        limit_iterations=limit_iterations,
        max_expansion_depth=max_expansion_depth,  # prevent overly-deep solutions
        prevent_repeat_mol_in_trees=prevent_repeat_mol_in_trees,  # original paper did this
        stop_on_first_solution=stop_on_first_solution,
    )
    alg = ReduceValueFunctionCallsRetroStar(
        and_node_cost_fn=and_node_cost_fn,
        value_function=value_function,
        or_node_cost_fn=or_node_cost_fn,
        **common_kwargs,
    )

    # Do search
    logger.info(f"Start search with {name}")
    # min_soln_times: list[tuple[float, ...]] = []
    if use_tqdm:
        smiles_iter = tqdm(smiles_list)
    else:
        smiles_iter = smiles_list

    # output_graph_dict = {}
    soln_time_dict = {}
    # routes_dict = {}
    # final_num_rxn_model_calls_dict = {}
    # final_num_value_function_calls_dict = {}
    # num_different_routes_dict = {}

    for i, smiles in enumerate(smiles_iter):
        logger.debug(f"Start search {i}/{len(smiles_list)}. SMILES: {smiles}")
        this_soln_times = list()
        # if isinstance(value_function, ConstantMolEvaluator):
        #     pass
        # else:
        alg.reset()
        output_graph, _ = alg.run_from_mol(Molecule(smiles))

        # Analyze solution time
        for node in output_graph.nodes():
            node.data["analysis_time"] = node.data["num_calls_rxn_model"]
        soln_time = get_first_solution_time(output_graph)
        this_soln_times.append(soln_time)

        # # Analyze number of routes
        # MAX_ROUTES = 10000
        # routes = list(iter_routes_cost_order(output_graph, MAX_ROUTES))

        # if alg.reaction_model.num_calls() < limit_rxn_model_calls:
        #     note = " (NOTE: this was less than the maximum budget)"
        # else:
        #     note = ""
        # logger.debug(
        #     f"Done {name}: nodes={len(output_graph)}, solution time = {soln_time}, "
        #     f"num routes = {len(routes)} (capped at {MAX_ROUTES}), "
        #     f"final num rxn model calls = {alg.reaction_model.num_calls()}{note}, "
        #     f"final num value model calls = {alg.value_function.num_calls}."
        # )

        # # Analyze route diversity
        # if (len(routes) > 0) & route_div:
        #     route_objects = [output_graph.to_synthesis_graph(nodes) for nodes in routes]
        #     packing_set = diversity.estimate_packing_number(
        #         routes=route_objects,
        #         distance_metric=diversity.reaction_jaccard_distance,
        #         radius=0.999,  # because comparison is > not >=
        #     )
        #     logger.debug((f"number of distinct routes = {len(packing_set)}"))
        # else:
        #     packing_set = []

        # Save results
        soln_time_dict.update({smiles: soln_time})
        # final_num_rxn_model_calls_dict.update({smiles: alg.reaction_model.num_calls()})
        # final_num_value_function_calls_dict.update(
        #     {smiles: alg.value_function.num_calls}
        # )
        # num_different_routes_dict.update({smiles: len(packing_set)})
        # output_graph_dict.update({smiles: output_graph})
        # routes_dict.update({smiles: routes})

    return SearchResult(
        name=name,
        soln_time_dict=soln_time_dict,
        # num_different_routes_dict=num_different_routes_dict,
        # final_num_rxn_model_calls_dict=final_num_rxn_model_calls_dict,
        # final_num_value_function_calls_dict=final_num_value_function_calls_dict,
        # output_graph_dict=output_graph_dict,
        # routes_dict=routes_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        # default="Guacamol_100_mid_hard_to_solve"
        # help="Text file with SMILES to run search on."
        # " One SMILES per line, no header.",
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
    parser.add_argument(
        "--limit_iterations",
        type=int,
        default=10_000,  # 2000
        help="Maximum number of algorithm iterations.",
    )
    parser.add_argument(
        "--limit_rxn_model_calls",
        type=int,
        default=1000,  # 500
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
        "--max_expansion_depth",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max_num_templates",
        type=int,
        default=50,  # 50
    )
    parser.add_argument(
        "--prevent_repeat_mol_in_trees",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--rxn_model",
        type=str,
        default="PAROUTES",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="PAROUTES",
    )
    # parser.add_argument(
    #     "--and_node_cost_fn",
    #     type=str,
    #     default="PAROUTES",
    # )
    parser.add_argument(
        "--or_node_cost_fn",
        type=str,
        default="MOL_PURCHASABLE",
    )
    parser.add_argument(
        "--stop_on_first_solution",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    # Create eventid
    # eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
    if args.input_file=="Guacamol_100_hardest_to_solve":
        eventid = '202306-3017-5132-d4cd54bf-e5a4-44c5-82af-c629a3692d87_HARDEST_COST1'
    if args.input_file=="Guacamol_100_hard_to_solve":
        eventid = "202307-0320-4900-d4c27728-a5aa-4177-8681-268a17c3d208_HARD_COST1"
    elif args.input_file=="Guacamol_100_mid_hard_to_solve":
        eventid = '202307-0620-3725-cc7b1f07-14cd-47e8-9d40-f5b2f358fa28_MID_HARD_COST1'
    elif args.input_file=="Guacamol_100_mid_to_solve":
        eventid = 'MID_COST1'
    elif args.input_file=="Guacamol_100_mid_easy_to_solve":
        eventid = 'MID_EASY_COST1'
    output_folder = f"CompareTanimotoLearnt/{eventid}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Logging
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

    # # Reduce logging from DeepChem
    # dc_log.setLevel(logging.DEBUG)

    # Read input SMILES
    with open(f"Data/{args.input_file}.txt") as f:
        test_smiles = [line.strip() for line in f.readlines()]

    if args.limit_num_smiles is not None:
        test_smiles = test_smiles[: args.limit_num_smiles]

    # if args.and_node_cost_fn == "PAROUTES":
    #     and_node_cost_fn = PaRoutesRxnCost()
    # else:
    #     raise NotImplementedError(f"and_node_cost_fn: {args.and_node_cost_fn}")

    if args.or_node_cost_fn == "MOL_PURCHASABLE":
        or_node_cost_fn = MolIsPurchasableCost()
    else:
        raise NotImplementedError(f"or_node_cost_fn: {args.or_node_cost_fn}")

    if args.inventory == "PAROUTES":
        inventory = PaRoutesInventory(n=args.paroutes_n)
    else:
        raise NotImplementedError(f"inventory: {args.inventory}")

    if args.rxn_model == "PAROUTES":
        rxn_model = PaRoutesModel(max_num_templates=args.max_num_templates)
    else:
        raise NotImplementedError(f"rxn_model: {args.rxn_model}")

    route_div = False

    # Load embedding model
    # emb_model_input_folder = {}
    # for embedding_model_to_use in args.embedding_models_to_use:
    #     emb_model_input_folder[embedding_model_to_use] = f'GraphRuns/{embedding_model_to_use}'
    # fnp_emb_model_input_folder = f'GraphRuns/{args.fnp_embedding_model_to_use}'
    # gnn_emb_model_input_folder = f'GraphRuns/{args.gnn_embedding_model_to_use}'

    value_fns_names = [
        # 'constant-0',
        # 'Tanimoto-distance',
        # 'Tanimoto-distance-TIMES01',
        # 'Tanimoto-distance-TIMES03',
        # 'Tanimoto-distance-TIMES10',
        # 'Tanimoto-distance-TIMES100',
        # 'Tanimoto-distance-TIMES1000',
        # 'Tanimoto-distance-EXP',
        # 'Tanimoto-distance-SQRT',
        # 'Tanimoto-distance-NUM_NEIGHBORS_TO_1',
        "Embedding-from-fingerprints",
        # "Embedding-from-fingerprints-TIMES10",
        # "Embedding-from-fingerprints-TIMES100",
        # "Embedding-from-fingerprints-TIMES1000",
        # "Embedding-from-fingerprints-TIMES10000",
        "Embedding-from-gnn",
        # "Embedding-from-gnn-TIMES10",
        # "Embedding-from-gnn-TIMES100",
        # "Embedding-from-gnn-TIMES1000",
        # "Embedding-from-gnn-TIMES10000",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fingerprint model
    # model_fnps, config_fnps = load_embedding_model_from_pickle(
    #     experiment_name=args.fnp_embedding_model_to_use
    # )
    model_fnps, config_fnps = load_embedding_model_from_checkpoint(
        experiment_name=args.fnp_embedding_model_to_use,
        device=device,
        checkpoint_name="epoch_11_checkpoint",
    )

    # Graph model
    # model_gnn, config_gnn = load_embedding_model_from_pickle(
    #     experiment_name=args.gnn_embedding_model_to_use,
    #     # model_name="epoch_51_checkpoint",
    # )
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

    logger.info(f"Start experiment {eventid}")
    args_string = ""
    for attr in dir(args):
        if not callable(getattr(args, attr)) and not attr.startswith("__"):
            args_string = args_string + "\n" + (f"{attr}: {getattr(args, attr)}")
    logger.info(f"Args: {args_string}")
    logger.info(f"dim_test: {len(test_smiles)}")

    result = {}
    for name, fn in value_fns:
        if name=='constant-0':
            and_node_cost_fn = PaRoutesRxnCost()
        else:
            and_node_cost_fn = ConstantNodeEvaluator(1.0)
        alg_result = run_algorithm(
            name=name,
            smiles_list=test_smiles,
            value_function=fn,
            rxn_model=rxn_model,
            inventory=inventory,
            and_node_cost_fn=and_node_cost_fn,
            or_node_cost_fn=or_node_cost_fn,
            max_expansion_depth=args.max_expansion_depth,
            prevent_repeat_mol_in_trees=args.prevent_repeat_mol_in_trees,
            use_tqdm=True,
            limit_rxn_model_calls=args.limit_rxn_model_calls,
            limit_iterations=args.limit_iterations,
            logger=logger,
            stop_on_first_solution=args.stop_on_first_solution,
        )
        result[name] = alg_result

        # Save pickle
        with open(f"{output_folder}/result_{name}.pickle", "wb") as handle:
            pickle.dump(alg_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
