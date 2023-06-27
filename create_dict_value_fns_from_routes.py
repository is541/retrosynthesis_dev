"""
"""
from __future__ import annotations

import pickle
import os
from tqdm.auto import tqdm
from paroutes import PaRoutesInventory
from rdkit.Chem import DataStructs, AllChem
from value_functions import initialize_value_functions

import numpy as np
import pandas as pd


def fingerprint_from_smiles(mol_smiles):
    return AllChem.GetMorganFingerprint(AllChem.MolFromSmiles(mol_smiles), radius=3)


if __name__ == "__main__":
    dataset_str = 'guacamol'
    # PAROUTES or GUACAMOL
    if dataset_str=='paroutes':
        run_id = "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7"
    elif dataset_str=='guacamol':
        run_id = "Guacamol_combined"
        
    input_file = f"Runs/{run_id}/routes_df.csv"
    routes_df = pd.read_csv(input_file)

    # output_file_routes = f"Runs/{run_id}/targ_routes.pickle"
    output_file_distances = f"Runs/{run_id}/targ_to_purch_distances.pickle"

    # Inventory
    inventory = PaRoutesInventory(n=5)
    purch_smiles = [mol.smiles for mol in inventory.purchasable_mols()]
    # len(purch_smiles)

    target_smiles = routes_df["target_smiles"].unique()

    # Save data dict with distances and rank (for every target),
    # one with sets of purchasable molecules, plus distance and rank (for every target)

    # Create distances dfs
    distances_df_dict = {}
    inventory=PaRoutesInventory(n=5)
    
    value_fns_names = [
        'Tanimoto-distance',
        "Embedding-from-fingerprints",
    ]
    
    value_fns = initialize_value_functions(value_fns_names, inventory)

    for target_smiles in target_smiles:
        for value_function_name, value_function in tqdm(value_fns):
            if value_function_name == 'Tanimoto-distance':
                purch_target_distance = 1 - value_function.get_similarity_with_purchasable_molecules(target_smiles)
                distance_tanimoto_df = pd.DataFrame(
                    {
                        "smiles": purch_smiles,
                        "Tanimoto_distance_from_target": purch_target_distance,
                    }
                )
                distance_tanimoto_df_sorted = distance_tanimoto_df.sort_values(
                    ["Tanimoto_distance_to_target", "smiles"], ascending=True
                ).reset_index(drop=True)
                distance_tanimoto_df_sorted["Tanimoto_distance_to_target_rank"] = distance_tanimoto_df_sorted.index + 1
                
            elif value_function_name == 'Embedding-from-fingerprints':
                purch_target_distance = value_function.get_distances_to_purchasable_molecules(target_smiles)
                distance_fnps_df = pd.DataFrame(
                    {
                        "smiles": purch_smiles,
                        "Fnps_distance_from_target": purch_target_distance,
                    }
                )
                distance_fnps_df_sorted = distance_fnps_df.sort_values(
                    ["Fnps_distance_to_target", "smiles"], ascending=True
                ).reset_index(drop=True)
                distance_fnps_df_sorted["Fnps_distance_to_target_rank"] = distance_fnps_df_sorted.index + 1
                
                
            distance_df = pd.merge(distance_tanimoto_df,distance_fnps_df, on='smiles', how='outer') 
            
            distances_df_dict[target_smiles] = distance_df
            
        with open(output_file_distances, "wb") as handle:
            pickle.dump(distances_df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # data_dict[name][value_function_name] = data_dict[name]['smiles'].map(smiles_value_fn_dict)
                
        ##### OLD CODE
        # target_fingerprint = fingerprint_from_smiles(target)
        # purch_fingerprints = list(map(fingerprint_from_smiles, purch_smiles))
        # purch_target_distance = [
        #     1 - sim
        #     for sim in DataStructs.BulkTanimotoSimilarity(
        #         target_fingerprint, purch_fingerprints
        #     )
        # ]
        # distance_df = pd.DataFrame(
        #     {
        #         "smiles": purch_smiles,
        #         "Tanimoto_distance_from_target": purch_target_distance,
        #     }
        # )

        # # Add rank
        # distance_df_sorted = distance_df.sort_values(
        #     ["Tanimoto_distance_from_target", "smiles"], ascending=True
        # ).reset_index(drop=True)
        # distance_df_sorted["distance_to_target_rank"] = distance_df_sorted.index + 1

        # distances_df_dict[target] = distance_df_sorted
        ##### END OLD CODE
        
        # if add_distances_to_routes_dict:
        #     routes_data_dict[target] = {}
        #     routes_dict_file = f"Runs/{run_id}/targ_routes.pickle"
        #     target_route_df = # Load from routes_dict_file

        #     target_route_df = pd.merge(
        #         target_route_df, distance_df_sorted, how="left", on="smiles"
        #     )

        #     target_mask = target_route_df["smiles"] == target
        #     target_route_df.loc[target_mask, "label"] = "Target"
        #     target_route_df.loc[target_mask, "Tanimoto_distance_from_target"] = 0
            

        #     with open(output_file_routes_updated, "wb") as handle:
        #         pickle.dump(routes_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open(output_file_distances, "wb") as handle:
        #     pickle.dump(distances_df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
