"""Basic code for nearest-neighbour value functions."""
from __future__ import annotations

from enum import Enum

import numpy as np
from rdkit.Chem import DataStructs, AllChem

from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.mol_inventory import ExplicitMolInventory
from syntheseus.search.node_evaluation.common import ConstantNodeEvaluator
from rdkit import Chem

import torch
import torch.nn.functional as F

# from Users.ilariasartori.syntheseus.search.graph.and_or import OrNode


class FakeMol:
    def __init__(
        self,
        smiles: str,
    ):
        super().__init__()
        self.smiles = smiles


class FakeOrNode:
    def __init__(
        self,
        smiles: str,
    ):
        super().__init__()
        self.mol = FakeMol(smiles)


class DistanceToCost(Enum):
    NOTHING = 0
    EXP = 1
    SQRT = 2
    TIMES10 = 3
    TIMES100 = 4
    TIMES1000 = 5
    TIMES10000 = 6
    NUM_NEIGHBORS_TO_1 = 7


class TanimotoNNCostEstimator(NoCacheNodeEvaluator):
    """Estimates cost of a node using Tanimoto distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_to_cost = distance_to_cost
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

    def get_fingerprint(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprint(mol, radius=3)

    def _set_fingerprints(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        self._fps = list(map(self.get_fingerprint, mols))

    def find_min_num_elem_summing_to_threshold(self, array, threshold):
        # Sort the array in ascending order
        sorted_array = np.sort(array)[::-1]

        # Calculate the cumulative sum of the sorted array
        cum_sum = np.cumsum(sorted_array)

        # Find the index where the cumulative sum exceeds threshold
        index = np.searchsorted(cum_sum, threshold)

        # Check if a subset of elements sums up to more than threshold
        if index < len(array):
            return index + 1  # Add 1 to account for 0-based indexing

        # If no subset of elements sums up to more than threshold
        return len(array)  # -1

    def get_similarity_with_purchasable_molecules(self, smiles: str):
        fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
        tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp_query, self._fps)
        return tanimoto_sims

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        tanimoto_sims = self.get_similarity_with_purchasable_molecules(smiles)
        if self.distance_to_cost == DistanceToCost.NUM_NEIGHBORS_TO_1:
            return 1 - self.find_min_num_elem_summing_to_threshold(
                array=tanimoto_sims, threshold=1
            ) / len(tanimoto_sims)
        else:
            return 1 - max(tanimoto_sims)

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0, f"Negative distance: {np.min(nn_dists)} "

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.SQRT:
            values = np.sqrt(nn_dists)
        elif self.distance_to_cost == DistanceToCost.TIMES10:
            values = 10.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES100:
            values = 100.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES1000:
            values = 1000.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.NUM_NEIGHBORS_TO_1:
            values = nn_dists
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)

    def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
        """Returns a dictionary of {molecule_smiles:molecule_value}."""
        values = self._evaluate_nodes(
            nodes=[FakeOrNode(smiles) for smiles in molecules_smiles]
        )
        return {k: v for k, v in zip(molecules_smiles, values)}


class Emb_from_fingerprints_NNCostEstimator(NoCacheNodeEvaluator):
    """Estimates cost of a node using Fingerprints-based Embeddings distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        model,
        distance_type,
        **kwargs,
    ):
        #         print('Stat initialization Emb')
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()

        self.distance_to_cost = distance_to_cost
        self.distance_type = distance_type
        self._set_fingerprints_vect(
            [mol.smiles for mol in inventory.purchasable_mols()]
        )
        with torch.no_grad():
            self.emb_purch_molecules = torch.stack(
                [
                    self.model(torch.tensor(fingerprint, dtype=torch.double))
                    for fingerprint in self._fps
                ],
                dim=0,
            )

    #         print('End initialization Emb')

    def get_fingerprint_vect(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)

    def _set_fingerprints_vect(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        self._fps = list(map(self.get_fingerprint_vect, mols))

    def compute_embedding_from_fingerprint(self, mol_fingerprints):
        #         self.model.eval()

        with torch.no_grad():
            output = self.model(torch.tensor(mol_fingerprints, dtype=torch.double))
        #             if isinstance(mol_fingerprints, list):
        #                 output =
        #             else:

        return output

    def embedding_distance(self, emb_1, emb_2):
        if self.distance_type == "Euclidean":
            # Compute Euclidean distance
            euclidean_distance = torch.norm(emb_1 - emb_2, dim=1)
            return euclidean_distance
        elif self.distance_type == "cosine":
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(emb_1, emb_2, dim=1)
            #             print(cosine_similarity)
            #             cosine_distance = 1 - cosine_similarity
            cosine_distance = torch.clamp(1 - cosine_similarity, min=0, max=1)
            #             print(cosine_distance)

            #             if np.min(cosine_distance)< 0 or np.max(cosine_distance) > 1:
            #                 if abs(cosine_distance - 0) < 1e-10:
            #                     return 0
            #                 elif abs(cosine_distance - 1) < 1e-10:
            #                     return 1
            #                 else:
            #                     raise ValueError(f"Cosine distance not between 0 and 1: Min: {np.min(cosine_distance)}, Max:{np.max(cosine_distance)}")
            return cosine_distance
        else:
            # Raise error for unsupported distance type
            raise NotImplementedError(
                f"Distance type '{self.distance_type}' is not implemented."
            )

    def get_distances_to_purchasable_molecules(self, smiles: str):
        fp_target = self.get_fingerprint_vect(
            AllChem.MolFromSmiles(smiles)
        )  # Target fingerprint
        emb_target = self.compute_embedding_from_fingerprint(
            fp_target
        )  # Target embedding

        #         emb_purch_molecules = self.compute_embedding_from_fingerprint(self._fps)  # Purchasable molecules embeddings
        #         print('Embedded purchasable molecules')

        # Euclidean (or cosine) distance between embeddings
        distances = self.embedding_distance(emb_target, self.emb_purch_molecules)
        return distances

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        distances = self.get_distances_to_purchasable_molecules(smiles)
        return torch.min(distances).item()

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0, f"Negative distance: {np.min(nn_dists)} "

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.SQRT:
            values = np.sqrt(nn_dists)
        elif self.distance_to_cost == DistanceToCost.TIMES10:
            values = 10.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES100:
            values = 100.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES1000:
            values = 1000.0 * nn_dists
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)

    def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
        """Returns a dictionary of {molecule_smiles:molecule_value}."""
        values = self._evaluate_nodes(
            nodes=[FakeOrNode(smiles) for smiles in molecules_smiles]
        )
        return {k: v for k, v in zip(molecules_smiles, values)}


class Emb_from_gnn_NNCostEstimator(NoCacheNodeEvaluator):
    """Estimates cost of a node using Gnn-based Embeddings distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        model,
        distance_type,
        featurizer,
        device,
        **kwargs,
    ):
        print("Stat initialization Emb")
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()

        self.distance_to_cost = distance_to_cost
        self.distance_type = distance_type
        self.featurizer = featurizer
        self.device = device

        self._purch_featurized = self._featurize_mols(
            [Chem.MolFromSmiles(mol.smiles) for mol in inventory.purchasable_mols()]
        )
        # self._set_dim_feat_vect()
        # self._set_fingerprints_vect([mol.smiles for mol in inventory.purchasable_mols()])
        with torch.no_grad():
            self.emb_purch_molecules = torch.stack(
                [
                    self.compute_embedding_from_feature_vect(purch_featurized)
                    for purch_featurized in self._purch_featurized
                ],
                dim=0,
            )
        print("End initialization Emb")

    def _featurize_mols(self, mols):
        return self.featurizer.featurize(mols, log_every_n=np.nan)
    
    # def _set_dim_feat_vect(self):
    #     for purch_featurized in self._purch_featurized:
    #         if isinstance(purch_featurized, np.ndarray):
    #             pass
    #         else:
    #             self.dim_feat_vect = purch_featurized.node_features.shape[1]
    #             break
        

    # def get_fingerprint_vect(self, mol: AllChem.Mol):
    #     return AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)

    # def _set_fingerprints_vect(self, smiles_list: list[str]) -> None:
    #     """Initialize fingerprint cache."""
    #     mols = list(map(AllChem.MolFromSmiles, smiles_list))
    #     assert None not in mols, "Invalid SMILES encountered."
    #     self._fps = list(map(self.get_fingerprint_vect, mols))

    def compute_embedding_from_feature_vect(self, mol_feat_vect):
        with torch.no_grad():
            if isinstance(mol_feat_vect, np.ndarray):
                output = torch.zeros(self.model.output_dim, dtype=torch.double)
            else:
                node_features = torch.tensor(
                    mol_feat_vect.node_features, dtype=torch.double
                ).to(self.device)
                edge_index = torch.tensor(mol_feat_vect.edge_index, dtype=torch.long).to(
                    self.device
                )
                output = self.model(node_features, edge_index)
                
        return output

    def embedding_distance(self, emb_1, emb_2):
        if self.distance_type == "Euclidean":
            # Compute Euclidean distance
            euclidean_distance = torch.norm(emb_1 - emb_2, dim=1)
            return euclidean_distance
        elif self.distance_type == "cosine":
            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(emb_1, emb_2, dim=1)
            cosine_distance = torch.clamp(1 - cosine_similarity, min=0, max=1)
            return cosine_distance
        else:
            # Raise error for unsupported distance type
            raise NotImplementedError(
                f"Distance type '{self.distance_type}' is not implemented."
            )

    def get_distances_to_purchasable_molecules(self, smiles: str):
        feat_target = self._featurize_mols(
            Chem.MolFromSmiles(smiles)
        )  # Target feature vector
        emb_target = self.compute_embedding_from_feature_vect(
            feat_target
        )  # Target embedding

        # Euclidean (or cosine) distance between embeddings
        distances = self.embedding_distance(emb_target, self.emb_purch_molecules)
        return distances

    def _get_nearest_neighbour_dist(self, smiles: str) -> float:
        distances = self.get_distances_to_purchasable_molecules(smiles)
        return torch.min(distances).item()

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dist(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0, f"Negative distance: {np.min(nn_dists)} "

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            values = nn_dists
        elif self.distance_to_cost == DistanceToCost.EXP:
            values = np.exp(nn_dists) - 1
        elif self.distance_to_cost == DistanceToCost.SQRT:
            values = np.sqrt(nn_dists)
        elif self.distance_to_cost == DistanceToCost.TIMES10:
            values = 10.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES100:
            values = 100.0 * nn_dists
        elif self.distance_to_cost == DistanceToCost.TIMES1000:
            values = 1000.0 * nn_dists
        else:
            raise NotImplementedError(self.distance_to_cost)

        return list(values)

    def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
        """Returns a dictionary of {molecule_smiles:molecule_value}."""
        values = self._evaluate_nodes(
            nodes=[FakeOrNode(smiles) for smiles in molecules_smiles]
        )
        return {k: v for k, v in zip(molecules_smiles, values)}


class ConstantMolEvaluator(NoCacheNodeEvaluator):
    # class ConstantMolEvaluator(ConstantNodeEvaluator):
    def __init__(
        self,
        constant_value: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # super().__init__(constant_value, **kwargs)
        self.constant_value = constant_value

    def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
        return {k: self.constant_value for k in molecules_smiles}

    def _evaluate_nodes(self, nodes, graph=None):
        return [self.constant_value] * len(nodes)


labelalias = {
    "constant-0": "constant-0",
    "Tanimoto-distance": "Tanimoto",
    "Tanimoto-distance-TIMES10": "Tanimoto * 10",
    "Tanimoto-distance-TIMES100": "Tanimoto * 100",
    "Tanimoto-distance-EXP": "Tanimoto exp",
    "Tanimoto-distance-SQRT": "Tanimoto sqrt",
    "Tanimoto-distance-NUM_NEIGHBORS_TO_1": "Tanimoto neighb to 1",
    "Embedding-from-fingerprints": "Emb fnps",
    "Embedding-from-fingerprints-TIMES10": "Emb fnps * 10",
    "Embedding-from-fingerprints-TIMES100": "Emb_fnps * 100",
    "Embedding-from-fingerprints-TIMES1000": "Emb_fnps * 1000",
    "Embedding-from-fingerprints-TIMES10000": "Emb_fnps * 10000",
    "Embedding-from-gnn": "Emb gnn",
}


def initialize_value_functions(
    value_fns_names,
    inventory,
    model_fnps=None,
    distance_type_fnps=None,
    model_gnn=None,
    distance_type_gnn=None,
    featurizer_gnn=None,
    device=None,
):
    value_fns = []
    for value_fns_name in value_fns_names:
        if value_fns_name == "constant-0":
            value_fns.append(("constant-0", ConstantMolEvaluator(0.0)))
        elif value_fns_name == "Tanimoto-distance":
            value_fns.append(
                (
                    "Tanimoto-distance",
                    TanimotoNNCostEstimator(
                        inventory=inventory, distance_to_cost=DistanceToCost.NOTHING
                    ),
                )
            )
        elif value_fns_name == "Tanimoto-distance-TIMES10":
            value_fns.append(
                (
                    "Tanimoto-distance-TIMES10",
                    TanimotoNNCostEstimator(
                        inventory=inventory, distance_to_cost=DistanceToCost.TIMES10
                    ),
                )
            )
        elif value_fns_name == "Tanimoto-distance-TIMES100":
            value_fns.append(
                (
                    "Tanimoto-distance-TIMES100",
                    TanimotoNNCostEstimator(
                        inventory=inventory, distance_to_cost=DistanceToCost.TIMES100
                    ),
                )
            )
        elif value_fns_name == "Tanimoto-distance-EXP":
            value_fns.append(
                (
                    "Tanimoto-distance-EXP",
                    TanimotoNNCostEstimator(
                        inventory=inventory, distance_to_cost=DistanceToCost.EXP
                    ),
                )
            )
        elif value_fns_name == "Tanimoto-distance-SQRT":
            value_fns.append(
                (
                    "Tanimoto-distance-SQRT",
                    TanimotoNNCostEstimator(
                        inventory=inventory, distance_to_cost=DistanceToCost.SQRT
                    ),
                )
            )
        elif value_fns_name == "Tanimoto-distance-NUM_NEIGHBORS_TO_1":
            value_fns.append(
                (
                    "Tanimoto-distance-NUM_NEIGHBORS_TO_1",
                    TanimotoNNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.NUM_NEIGHBORS_TO_1,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-fingerprints":
            if model_fnps is None or distance_type_fnps is None:
                raise ValueError(
                    "Both model_fnps and distance_type_fnps must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints",
                    Emb_from_fingerprints_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.NOTHING,
                        model=model_fnps,
                        distance_type=distance_type_fnps,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-fingerprints-TIMES10":
            if model_fnps is None or distance_type_fnps is None:
                raise ValueError(
                    "Both model_fnps and distance_type_fnps must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints-TIMES10",
                    Emb_from_fingerprints_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.TIMES10,
                        model=model_fnps,
                        distance_type=distance_type_fnps,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-fingerprints-TIMES100":
            if model_fnps is None or distance_type_fnps is None:
                raise ValueError(
                    "Both model_fnps and distance_type_fnps must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints-TIMES100",
                    Emb_from_fingerprints_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.TIMES100,
                        model=model_fnps,
                        distance_type=distance_type_fnps,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-fingerprints-TIMES1000":
            if model_fnps is None or distance_type_fnps is None:
                raise ValueError(
                    "Both model_fnps and distance_type_fnps must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints-TIMES1000",
                    Emb_from_fingerprints_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.TIMES1000,
                        model=model_fnps,
                        distance_type=distance_type_fnps,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-fingerprints-TIMES10000":
            if model_fnps is None or distance_type_fnps is None:
                raise ValueError(
                    "Both model_fnps and distance_type_fnps must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints-TIMES10000",
                    Emb_from_fingerprints_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.TIMES10000,
                        model=model_fnps,
                        distance_type=distance_type_fnps,
                    ),
                )
            )
        elif value_fns_name == "Embedding-from-gnn":
            if model_gnn is None or (
                distance_type_gnn is None or (featurizer_gnn is None or device is None)
            ):
                raise ValueError(
                    "model_gnn, distance_type_gnn, featurizer_gnn and device must be provided."
                )
            value_fns.append(
                (
                    "Embedding-from-fingerprints",
                    Emb_from_gnn_NNCostEstimator(
                        inventory=inventory,
                        distance_to_cost=DistanceToCost.NOTHING,
                        model=model_gnn,
                        distance_type=distance_type_gnn,
                        featurizer=featurizer_gnn,
                        device=device,
                    ),
                )
            )

    return value_fns
