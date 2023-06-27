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

from syntheseus.search import visualization

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
            os.makedirs(f"{output_folder}/graph_pdfs")
            
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
        
    test_smiles = [
        # "CCCCCCCCCCCCN(C)C",
        # "NNC(c1ccccc1)C(O)c1ccccc1"
        "NC(=O)c1ccc2nc(NC3CCC(O)CC3)c3nccn3c2c1",
        # "O=C1c2ccccc2NC(c2cnc3ccccc3n2)N1O",
        # "O=C(Nc1ccc(C(=O)N2CCOCC2)cc1)Nc1ccc(C(F)(F)F)cc1",
        "O=C(O)c1ccccc1C(=O)Nc1cc(C(F)(F)F)ccc1N1CCCC1",
        # "Cc1cc(-c2ncc(C(F)(F)F)[nH]2)ccc1OCC(O)CNC(C)(C)C",
        # "Fc1ccc(-c2nnc3ccc(Sc4cccc(C(F)(F)F)c4)nn23)cc1",
        # "O=C(C=CC=Cc1ccc2c(c1)OCO2)Nc1c(F)cc(F)cc1F",
        "CC1=NC2(N=C1N)c1cc(-c3cncc(C(F)(F)F)c3)ccc1CC21CCC(OC(F)F)CC1",
        # "O=C(Nc1ccc(Br)cc1F)c1cc2nc(-c3ccco3)cc(C(F)(F)Cl)n2n1",
        # "Cc1cc(=O)oc2cc(NC(=O)CCCCCCNC(=O)c3cc(F)cc(F)c3)ccc12",
        # "CCc1ncc(C(CN=C(O)c2cccc(Cl)c2Cl)N2CCC(F)(F)CC2)cn1"
        # "Fc1cc(C(=S)Nc2ccccc2NC(=S)c2cc(F)c(F)c(F)c2)cc(F)c1F",
        # "Cn1c(SCCCN2CCOC(c3ccc(C(F)(F)F)cc3)C2)nnc1-c1ccc(-n2cnnc2)cc1",
        # "Cc1nc2cccc(-c3cc4c([nH]3)CCN=C4O)c2nc1NC1CC(F)(F)C1",
        # "CC(C)C1CCC(N2CCC(O)(c3cccc(C(F)(F)F)c3)CC2)CC1",
        # "CC1(c2cccc(NC(=O)C3CCC(F)(F)CC3)c2)COCC(N)=N1",
        # "Cc1nc(C(F)(F)F)cn1-c1ccc(NCc2c(F)cccc2F)nc1",
        # "CN(C)CCN(C)C(=O)c1ccccc1-c1ccc2c3n[nH]cc3c(=O)n(CC(F)(F)F)c2c1",
        "CS(=O)(=O)CCCn1c(=N)sc2cc(OC(F)(F)F)ccc21", # Maybe
        # "CCCC(O)C(CNCC(C)C)NC(=O)CNC(=O)c1cc(C(F)(F)F)ccc1NC(=O)NC(C)C", 
        # "Cc1ccc(-c2nc3sc(C(F)(F)F)nn3c2-c2ccccc2)cc1",
        # "COc1ccc(CN2C(CC(=O)O)c3cc(C(F)(F)F)ccc3S2(=O)=O)cc1",
        "FC(F)(F)c1csc(Nc2cc3ccccc3cn2)n1", # Example for easy synthesis route (4)
        # "O=C(NCCc1ccc(O)c2ccc(O)cc12)c1ccc(OC(F)(F)F)cc1",
        # "O=C(Nc1ccc(C(F)(F)F)nc1)N1CCN(c2ccccc2C(F)(F)F)CC1",
        # "OCC1(NCc2cccc(-c3ccc(C(F)(F)F)cc3C(F)(F)F)n2)CCCC1",
        "Cc1nnc(SCC2=C(C(=O)O)N3C(=O)C(NC(=O)Cn4nc(C(F)F)c(Cl)c4C)C3SC2)s1", # Maybe (is it solvable?)
        # "O=C(CCSc1nc(-c2ccco2)cc(C(F)(F)F)n1)NCCc1ccccc1",
        # "N#Cc1cc(C(F)(F)F)ccc1N1CCC(N(c2ccc(C(F)(F)F)cc2)c2cccnc2)CC1",
        # "O=[N+]([O-])c1cn2c(n1)OCC(Oc1ccc(-c3ccc(C(F)(F)F)nc3)cc1)C2",
        # PAROUTES
        # "CCOc1cn(-c2cccc(C(F)(F)F)c2)nc(-c2ccnn2-c2ccccc2)c1=O",
        # "Cn1nc(-c2ccc(OC(F)(F)F)cc2)cc1CCOc1ccc2ccn(CC(=O)O)c2c1",
        # "O=c1n(Cc2ccc(C(F)(F)F)nc2O)nc2c(-c3ccncc3)c(-c3ccc(Cl)cc3)ccn12",
        # "COC(=O)c1cc(Cc2ccc(OC)cc2)c(C)c(Cl)c1OS(=O)(=O)C(F)(F)F",
        # "Cc1c2c(cc3[nH]c(-c4c(NC(C)Cc5c(F)c(F)cc(F)c5F)cc[nH]c4=O)nc13)C(=O)N(CCN(C)C)C2",
        # "CNC(=O)c1nn(-c2ccccc2)c(NC(=O)NCc2cc(COC)ccc2OC(F)(F)F)c1C",
        # "c1nc(-c2ccc(C(F)(F)F)cc2)sc1COc1ccc2c(Cl)cn(CC(=O)O)c2c1",
        # "Cc1cc(C2=NOC(c3cc(Cl)c(F)c(Cl)c3)(C(F)(F)F)C2)ccc1C(=O)NCc1ccc2c(c1)B(O)OC2(C)C",
        # "O=C(Cc1c(F)cc(Br)cc1F)Nc1ccc(OCCOCc2ccccc2)c(C(F)(F)F)c1",
        # "Cc1ccc(S(=O)(=O)OC[C@@](O)(CNC(=O)c2cnn(-c3ccc(F)cc3)c2N)C(F)(F)F)cc1",
        # "O=C(NC1CCN(CC(F)(F)F)CC1)c1ccc(-c2ccccc2F)nc1",
        # "Nc1ccc(C(F)(F)F)cc1NC(=O)C1CCC2(CC1)CC(=O)N(c1ccccc1)C2"
        # "CCCCCCC(Oc1ccc(C(=O)NCCC(=O)O)cc1F)c1ccc(-c2ccc(C(F)(F)F)cc2)c(C)c1",
        # "Cn1c(Nc2cc(CN)c(F)cc2Cl)nc2cc(Cl)c(N3CCC(C(F)(F)F)CC3)cc21",
        # "O=C(Cc1ccc(F)c(C(F)(F)F)c1)Nc1c(Cl)ccc2c(=O)n(CCO)ccc12",
        # "CC(C)(C)C(=O)NCc1ccc(Cl)c(C(=O)Nc2cccc3c(Oc4cccc(C(F)(F)F)c4)ncnc23)c1F",
        # "CC(C)(O)CNC(=O)c1cc(Cl)n2c(CC3CCC(F)(F)CC3)c(C(F)(F)F)nc2c1",
        # "Nc1nccn2c([C@@H]3CC[C@@H](CNC(=O)OCc4ccccc4)CC3)nc(I)c12",
        # "Cc1nc(C(C)(CC2CC2)NC(=O)c2cc(OCC(F)(F)F)c(Cl)cn2)no1",
        "CS(=O)(=O)Cc1cccc2c(C(c3ccc(Cl)cc3F)C3CC3C#N)c[nH]c12", # Example for building tree
        # "CCCN(CCC)c1ccc(N)c(-c2cc(C(=O)NCc3cccc(C(F)(F)F)c3)ccn2)c1",
        # "O=C(O)CCC(=O)O[C@@H]1CC[C@@H](Nc2ncc3nc(Nc4c(F)cccc4F)n(C4CCCC4)c3n2)CC1",
        # "C[C@H](NC(=O)Cc1cc(F)cc(F)c1)C(=O)N[C@@H]1C(=O)N(CC2CC2)c2ccccc2O[C@@H]1c1ccccc1",
        # "CC(C)(C)OC(=O)N1CCN(c2nc3ccc(Br)c(F)c3c3ncnn23)CC1",
        # "Cc1c(-c2c(Cl)cc(C(N)=O)c3[nH]c4cc(C(C)(C)O)ccc4c23)cccc1-n1c(=O)[nH]c2c(F)cc(F)cc2c1=O",
        # "CC(C)[C@H](C(=O)NC1CCC(F)(F)C1)N1Cc2c(F)cnc3[nH]cc(c23)C1=O",
        # "CC(C)N(CCCNC(=O)Nc1ccc(Cl)c(C(F)(F)F)c1)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O"
    ]
    test_smiles = [
        "FC(F)(F)c1csc(Nc2cc3ccccc3cn2)n1", # Example for easy synthesis route
        "CS(=O)(=O)Cc1cccc2c(C(c3ccc(Cl)cc3F)C3CC3C#N)c[nH]c12", # Example for building tree
    ]
    test_smiles = [
    "COc1c(N2CCN(C(=O)COC(C)=O)CC2)cc(Br)cc1C(C)(C)C", # Nice (1)
    "C=CC(OCOC)c1ccc(N(c2cc3oc(-c4ccc(F)cc4)c(C(=O)NC)c3cc2C2CC2)S(C)(=O)=O)cc1Cl", # USED - Route molecole easy all'inizio, complesse in profondità
#     "CC(C(=O)O)c1ccc2nc(N)oc2c1",
#     "Cc1ccc2ccccc2c1CC(C)Oc1ccccc1N",
#     "Cc1c(Oc2ccc(C(CO)CO)cc2)ncnc1OC1CCN(c2ncccn2)CC1",
#     "CCCCCCCCCCCCCCCCSCC(CBr)OC",
    "O=C(NCCc1ccc(Cl)cc1C1CC1)c1ccc(Oc2cc3c(cc2Cl)C(C(=O)O)CCO3)cc1", # USED - Nice (2)
    "N#Cc1ccccc1-c1cc(F)cc(-c2ccnc3nc(C(F)(F)F)ccc23)c1", # Nice (3)
    "Cc1cccc2c(Cl)ncc(C(=O)NCC3CCOCC3)c12", # Ok
#     "Cc1nc(C(C)(CS(C)(=O)=O)NC(=O)c2cc(O[C@@H](C)C(F)(F)F)c(C3CCC3)cn2)no1",
#     "CC(C)(CC1CCN(c2ccc([N+](=O)[O-])cn2)CC1)C(=O)O",
#     "FC(F)(F)c1ccc(COc2ccc(Br)cc2CC2CNC2)o1",
#     "CC(=O)NCCOc1ccc2c(c1)c(Cl)cn2S(=O)(=O)c1ccccc1",
#     "COc1ccc(Oc2ccccc2Cl)cc1C(C)C(=O)O",
#     "CC1(NC(=O)OC(C)(C)C)CCCN(c2c(N)cnc3c2CCC3)C1",
#     "CCN1CCN(C2(C(=O)N3C[C@H](S(=O)(=O)c4ccccc4Cl)C[C@H]3C(=O)NC3(C#N)CC3)CC2)CC1",
#     "Oc1ccc2nc(-c3ccc(-c4nn[nH]n4)cc3Cl)ccc2c1",
#     "CCOC(=O)Cc1ccc(OC)c(-c2ccc(NC(C)=O)cc2CN(CC)C(=O)OCc2ccccc2)c1",
#     "Cc1cccc(NC2(CO)CCN(Cc3ccccc3)CC2)c1",
    "CN(C)Cc1cc(C(C)(C)C)cc(Cl)n1", # Nice
#     "CCOC(=O)C(C)(CCCc1ccccc1)Cc1ccc(OCCN)cc1",
#     "CCCC(Oc1ccccc1Br)C(C)=O",
#     "CN(Cc1ccccc1)C(=O)c1cc(NC(=O)c2cc(F)c(F)cc2Cl)[nH]n1",
    "COC(=O)c1cc(N=Nc2ccc(OCCCCCCCCN=[N+]=[N-])cc2)ccc1O", # From 1 reaction exit 3 molecules
    ]
    
    # # Try with 3 templates
    # test_smiles = [
    #     "CC1=NC2(N=C1N)c1cc(-c3cncc(C(F)(F)F)c3)ccc1CC21CCC(OC(F)F)CC1",   
    # ]

    # Make reaction model, inventory, value functions, algorithm
    rxn_model = PaRoutesModel(max_num_templates=4)
    inventory = PaRoutesInventory(n=args.paroutes_n)
    value_function = ConstantNodeEvaluator(0.0)
    algorithm = ReduceValueFunctionCallsRetroStar(
        reaction_model=rxn_model,
        mol_inventory=inventory,
        limit_reaction_model_calls=args.limit_rxn_model_calls,
        limit_iterations=args.limit_iterations,
        max_expansion_depth=5,  # prevent overly-deep solutions
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
            print("Smiles index i: ", i)
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
                    
            # Plot output graph
            visualization.visualize_andor(output_graph, f"{output_folder}/graph_pdfs/mol_{i}_graph.pdf")

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
