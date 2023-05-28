"""Basic code for nearest-neighbour value functions."""
from __future__ import annotations

from enum import Enum
import functools
import heapq

import numpy as np
from rdkit.Chem import DataStructs, AllChem

from syntheseus.search.graph.and_or import OrNode
from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
from syntheseus.search.mol_inventory import ExplicitMolInventory


class DistanceToCost(Enum):
    NOTHING = 0
    EXP = 1


class TanimotoNNCostEstimator(NoCacheNodeEvaluator):
    """Estimates cost of a node using Tanimoto distance to purchasable molecules."""

    def __init__(
        self,
        inventory: ExplicitMolInventory,
        distance_to_cost: DistanceToCost,
        nearest_neighbour_cache_size: int = 10_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.distance_to_cost = distance_to_cost
        self.nearest_neighbour_cache_size = nearest_neighbour_cache_size
        self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

    def get_fingerprint(self, mol: AllChem.Mol):
        return AllChem.GetMorganFingerprint(mol, radius=3)

    def _set_fingerprints(self, smiles_list: list[str]) -> None:
        """Initialize fingerprint cache."""
        mols = list(map(AllChem.MolFromSmiles, smiles_list))
        assert None not in mols, "Invalid SMILES encountered."
        _fps = list(map(self.get_fingerprint, mols))

        @functools.lru_cache(maxsize=self.nearest_neighbour_cache_size)
        def _top_few_fp_sims(smiles: str):
            fp = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
            tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp, _fps)
            top_sims = heapq.nlargest(5, tanimoto_sims)
            return [1 - sim for sim in top_sims]

        self._get_nearest_neighbour_dists = _top_few_fp_sims

    def _evaluate_nodes(self, nodes: list[OrNode], graph=None) -> list[float]:
        if len(nodes) == 0:
            return []

        # Get distances to nearest neighbours
        nn_dists = np.asarray(
            [self._get_nearest_neighbour_dists(node.mol.smiles) for node in nodes]
        )
        assert np.min(nn_dists) >= 0  # ensure distances are valid

        # Function above returns top 5 distances,
        # but we base the value on just the top 1 neighbour
        top1_nn_dists = np.min(nn_dists, axis=1)

        # Turn into costs
        if self.distance_to_cost == DistanceToCost.NOTHING:
            return top1_nn_dists.tolist()
        elif self.distance_to_cost == DistanceToCost.EXP:
            return (np.exp(top1_nn_dists) - 1).tolist()
        else:
            raise NotImplementedError(self.distance_to_cost)
