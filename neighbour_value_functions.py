# """Basic code for nearest-neighbour value functions."""
# from __future__ import annotations

# from enum import Enum

# import numpy as np
# from rdkit.Chem import DataStructs, AllChem

# from syntheseus.search.graph.and_or import OrNode
# from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
# from syntheseus.search.mol_inventory import ExplicitMolInventory

# # from Users.ilariasartori.syntheseus.search.graph.and_or import OrNode


# class DistanceToCost(Enum):
#     NOTHING = 0
#     EXP = 1
#     SQRT = 2
#     TIMES10 = 3
#     TIMES100 = 4
#     NUM_NEIGHBORS_TO_1 = 5
#     NUM_NEIGHBORS_TO_1_TIMES1000 = 6


# class TanimotoNNCostEstimator():
#     """Estimates cost of a molecule using Tanimoto distance to purchasable molecules."""

#     def __init__(
#         self,
#         inventory: ExplicitMolInventory,
#         distance_to_cost: DistanceToCost,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.distance_to_cost = distance_to_cost
#         self._set_fingerprints([mol.smiles for mol in inventory.purchasable_mols()])

#     def get_fingerprint(self, mol: AllChem.Mol):
#         return AllChem.GetMorganFingerprint(mol, radius=3)

#     def _set_fingerprints(self, smiles_list: list[str]) -> None:
#         """Initialize fingerprint cache."""
#         mols = list(map(AllChem.MolFromSmiles, smiles_list))
#         assert None not in mols, "Invalid SMILES encountered."
#         self._fps = list(map(self.get_fingerprint, mols))
        
#     def find_min_num_elem_summing_to_threshold(self, array, threshold):
#         # Sort the array in ascending order
#         sorted_array = np.sort(array)[::-1]

#         # Calculate the cumulative sum of the sorted array
#         cum_sum = np.cumsum(sorted_array)

#         # Find the index where the cumulative sum exceeds threshold 
#         index = np.searchsorted(cum_sum, threshold)

#         # Check if a subset of elements sums up to more than threshold
#         if index < len(array):
#             return index + 1  # Add 1 to account for 0-based indexing

#         # If no subset of elements sums up to more than threshold
#         return len(array) #-1

#     def _get_nearest_neighbour_dist(self, smiles: str) -> float:
#         fp_query = self.get_fingerprint(AllChem.MolFromSmiles(smiles))
#         tanimoto_sims = DataStructs.BulkTanimotoSimilarity(fp_query, self._fps)
#         if self.distance_to_cost in [DistanceToCost.NUM_NEIGHBORS_TO_1, DistanceToCost.NUM_NEIGHBORS_TO_1_TIMES1000]:
#             return (self.find_min_num_elem_summing_to_threshold(array=tanimoto_sims,threshold=1)-1)/ len(tanimoto_sims)
#         else:
#             return 1 - max(tanimoto_sims)

#     def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
#         """Returns a dictionary of {molecule_smiles:molecule_value}."""
#         if len(molecules_smiles) == 0:
#             return {}

#         # Get distances to nearest neighbours
#         nn_dists = np.asarray(
#             [self._get_nearest_neighbour_dist(mol_smiles) for mol_smiles in molecules_smiles]
#         )
#         assert np.min(nn_dists) >= 0

#         # Turn into costs
#         if self.distance_to_cost == DistanceToCost.NOTHING:
#             values = nn_dists
#         elif self.distance_to_cost == DistanceToCost.EXP:
#             values = np.exp(nn_dists) - 1
#         elif self.distance_to_cost == DistanceToCost.SQRT:
#             values = np.sqrt(nn_dists) 
#         elif self.distance_to_cost == DistanceToCost.TIMES10:
#             values = 10.0*nn_dists
#         elif self.distance_to_cost == DistanceToCost.TIMES100:
#             values = 100.0*nn_dists
#         elif self.distance_to_cost == DistanceToCost.NUM_NEIGHBORS_TO_1:
#             values = nn_dists
#         elif self.distance_to_cost == DistanceToCost.NUM_NEIGHBORS_TO_1_TIMES1000:
#             values = 1000.0*nn_dists
#         else:
#             raise NotImplementedError(self.distance_to_cost)

#         return {k: v for k, v in zip(molecules_smiles, values)}
    
    
# class ConstantMolEvaluator():  
#     def __init__(
#         self,
#         constant_value: float,
#         **kwargs,
#     ):
#         self.constant_value=constant_value
    
#     def evaluate_molecules(self, molecules_smiles: list[str]) -> dict:
#         return {k: self.constant_value for k in molecules_smiles}
    