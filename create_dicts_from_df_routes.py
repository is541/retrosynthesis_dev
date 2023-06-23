"""
"""
from __future__ import annotations

import pickle
import os
from tqdm.auto import tqdm
from paroutes import PaRoutesInventory
from rdkit.Chem import DataStructs, AllChem

import numpy as np
import pandas as pd


def fingerprint_from_smiles(mol_smiles):
    return AllChem.GetMorganFingerprint(AllChem.MolFromSmiles(mol_smiles), radius=3)


if __name__ == "__main__":
    run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"

    input_file = f"Runs/{run_id}/routes_df.csv"
    routes_df = pd.read_csv(input_file)

    output_file_routes = f"Runs/{run_id}/targ_routes.pickle"
    output_file_distances = f"Runs/{run_id}/targ_to_purch_distances.pickle"

    # Inventory
    inventory = PaRoutesInventory(n=5)
    purch_smiles = [mol.smiles for mol in inventory.purchasable_mols()]
    # len(purch_smiles)

    target_smiles = routes_df["target_smiles"].unique()
    cols_to_keep_renamed = {
        "route_rank": "label",
        "intermediate_smiles": "smiles",
        "intermediate_depth": "depth",
    }

    # Save 2 separate data dict: one with distances and rank (for every target),
    # one with sets of purchasable molecules, plus distance and rank (for every target)

    # Create distances dfs
    distances_df_dict = {}
    routes_data_dict = {}
    for target in tqdm(target_smiles):
        # 1 - Distances dfs
        # Compute distances
        target_fingerprint = fingerprint_from_smiles(target)
        purch_fingerprints = list(map(fingerprint_from_smiles, purch_smiles))
        purch_target_distance = [
            1 - sim
            for sim in DataStructs.BulkTanimotoSimilarity(
                target_fingerprint, purch_fingerprints
            )
        ]
        distance_df = pd.DataFrame(
            {
                "smiles": purch_smiles,
                "Tanimoto_distance_from_target": purch_target_distance,
            }
        )

        # Add rank
        distance_df_sorted = distance_df.sort_values(
            ["Tanimoto_distance_from_target", "smiles"], ascending=True
        ).reset_index(drop=True)
        distance_df_sorted["distance_to_target_rank"] = distance_df_sorted.index + 1

        distances_df_dict[target] = distance_df_sorted

        # 2 - Routes df
        target_df = routes_df.loc[routes_df["target_smiles"] == target]
        routes_data_dict[target] = {}

        for route_rank in target_df["route_rank"].dropna().unique():
            target_route_df = target_df.loc[
                (
                    (target_df["intermediate_is_purchasable"])
                    | (target_df["intermediate_smiles"] == target)
                )
                & (target_df["route_rank"] == route_rank),
                cols_to_keep_renamed.keys(),
            ].drop_duplicates()
            route_name = "route_" + str(int(route_rank))
            target_route_df["route_rank"] = route_name
            #         target_route_df['route_rank'] = 'route_' + target_route_df['route_rank'].astype(int).astype(str)

            target_route_df = target_route_df.rename(columns=cols_to_keep_renamed)

            target_route_df = pd.merge(
                target_route_df, distance_df_sorted, how="left", on="smiles"
            )

            target_mask = target_route_df["smiles"] == target
            target_route_df.loc[target_mask, "label"] = "Target"
            target_route_df.loc[target_mask, "Tanimoto_distance_from_target"] = 0

            routes_data_dict[target].update({route_name: target_route_df})

        with open(output_file_routes, "wb") as handle:
            pickle.dump(routes_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_file_distances, "wb") as handle:
            pickle.dump(distances_df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
