"""
Script to train neural network to get molecules embeddings
"""
from __future__ import annotations

import argparse
import os
import pickle
import json
import random
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time

from tqdm.auto import tqdm
from paroutes import PaRoutesInventory, get_target_smiles
from embedding_model import (
    fingerprint_preprocess_input,
    gnn_preprocess_input,
    CustomDataset,
    collate_fn,
    # SampleData,
    fingerprint_vect_from_smiles,
    compute_embeddings,
    GNNModel,
    FingerprintModel,
    NTXentLoss,
    num_heavy_atoms
)
from paroutes import PaRoutesInventory
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import plotly.express as px
from rdkit import Chem
import deepchem as dc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, required=True, help="JSON file with configurations."
    )
    # parser.add_argument(
    #     "--save_preprocessed_data",
    #     type=bool,
    #     default=True,
    #     help="Whether to save the pickle of the preprocessed data",
    # )
    # parser.add_argument(
    #     "--load_from_preprocessed_data",
    #     type=bool,
    #     default=True,
    #     help="Whether to skip preprocessing and read from previously saved pickle",
    # )
    args = parser.parse_args()

    with open(f"{args.config_file}.json", "r") as f:
        config = json.load(f)

    # MODEL TO EVALUATE
    experiment_name = config["experiment_name"]  # gnn_0629
    checkpoint_folder = f"GraphRuns/{experiment_name}/"
    input_checkpoint_name = f"epoch_61_checkpoint.pth"
    
    
    # 1. PREPROCESS DATA
    # if not args.load_from_preprocessed_data:

    # Read routes data
    input_file_routes = f'Runs/{config["run_id"]}/targ_routes.pickle'
    # input_file_distances = f'Runs/{config["run_id"]}/targ_to_purch_distances.pickle'

    # Routes data
    with open(input_file_routes, "rb") as handle:
        targ_routes_dict = pickle.load(handle)

    # # Load distances data
    # with open(input_file_distances, 'rb') as handle:
    #     distances_dict = pickle.load(handle)

    # Inventory

    inventory = PaRoutesInventory(n=5)
    purch_smiles = [mol.smiles for mol in inventory.purchasable_mols()]
    # len(purch_smiles)

    # def num_heavy_atoms(mol):
    #     return Chem.rdchem.Mol.GetNumAtoms(mol, onlyExplicit=True)

    purch_mol_to_exclude = []
    purch_nr_heavy_atoms = {}
    for smiles in purch_smiles:
        nr_heavy_atoms = num_heavy_atoms(Chem.MolFromSmiles(smiles))
        if nr_heavy_atoms < 2:
            purch_mol_to_exclude = purch_mol_to_exclude + [smiles]
        purch_nr_heavy_atoms[smiles] = nr_heavy_atoms

    if config["run_id"] == "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7":
        all_targets = get_target_smiles(n=5)
    elif config["run_id"] == "Guacamol_combined":
        with open("Data/Guacamol/guacamol_v1_test_10ksample.txt", "r") as f:
            all_targets = [line.strip() for line in f.readlines()]

    targ_route_not_in_route_dict = {}
    for target in all_targets:
        targ_route_not_in_route_dict[target] = {}

        target_routes_dict = targ_routes_dict.get(target, "Target_Not_Solved")

        if target_routes_dict == "Target_Not_Solved":
            purch_in_route = []
        else:
            target_route_df = target_routes_dict["route_1"]
            purch_in_route = list(
                target_route_df.loc[target_route_df["label"] != "Target", "smiles"]
            )
        #         purch_in_route = [smiles for smiles in purch_in_route if smiles in purch_smiles]
        purch_not_in_route = [
            purch_smile
            for purch_smile in purch_smiles
            if purch_smile not in purch_in_route
        ]
        random.seed(config["seed"])

        if config["neg_sampling"] == "uniform":
            purch_not_in_route_sample = random.sample(
                purch_not_in_route, config["not_in_route_sample_size"]
            )
        elif config["neg_sampling"] == "...":
            pass
        else:
            raise NotImplementedError(f'{config["neg_sampling"]}')

        # Filter out molecules with only one atom (problems with featurizer)
        purch_in_route = [
            smiles for smiles in purch_in_route if smiles not in purch_mol_to_exclude
        ]
        purch_not_in_route_sample = [
            smiles
            for smiles in purch_not_in_route_sample
            if smiles not in purch_mol_to_exclude
        ]

        targ_route_not_in_route_dict[target]["positive_samples"] = purch_in_route
        targ_route_not_in_route_dict[target][
            "negative_samples"
        ] = purch_not_in_route_sample

    # Get a random sample of keys from targ_routes_dict
    if config["nr_sample_targets"] != -1:
        sample_targets = random.sample(
            list(targ_route_not_in_route_dict.keys()), config["nr_sample_targets"]
        )
    else:
        sample_targets = targ_route_not_in_route_dict
    # Create targ_routes_dict_sample with the sampled keys and their corresponding values
    targ_route_not_in_route_dict_sample = {
        target: targ_route_not_in_route_dict[target] for target in sample_targets
    }

    input_data = targ_route_not_in_route_dict_sample

    if config["model_type"] == "gnn":
        featurizer = dc.feat.MolGraphConvFeaturizer()

        purch_mols = [Chem.MolFromSmiles(smiles) for smiles in purch_smiles]
        purch_featurizer = featurizer.featurize(purch_mols)
        purch_featurizer_dict = dict(zip(purch_smiles, purch_featurizer))

        dataset = gnn_preprocess_input(
            input_data=input_data, 
            featurizer=featurizer, 
            featurizer_dict=purch_featurizer_dict,
            pos_sampling=config["pos_sampling"],
        )
        
    elif config["model_type"] == "fingerprints":
        purch_fingerprints = list(map(fingerprint_vect_from_smiles, purch_smiles))
        purch_fingerprints_dict = dict(zip(purch_smiles, purch_fingerprints))

        dataset = fingerprint_preprocess_input(
            input_data, 
            fingerprints_dict=purch_fingerprints_dict, 
            pos_sampling=config["pos_sampling"],
        )
        
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')


    # 2. TRAIN VALIDATION SPLIT
    validation_ratio = config["validation_ratio"]
    num_samples = len(dataset)
    num_val_samples = int(validation_ratio * num_samples)
    
    train_indices, val_indices = train_test_split(
        range(num_samples), test_size=num_val_samples, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=config["train_shuffle"],
        collate_fn=collate_fn,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=config["val_shuffle"],
        collate_fn=collate_fn,
    )
    
    # Batch size: The batch size determines the number of samples processed in each iteration during training or validation. In most cases, it is common to use the same batch size for both training and validation to maintain consistency. However, there are situations where you might choose a different batch size for validation. For instance, if memory constraints are more relaxed during validation, you can use a larger batch size to speed up evaluation.
    # Shuffle training data: Shuffling the training data before each epoch is beneficial because it helps the model see the data in different orders, reducing the risk of the model learning patterns specific to the order of the data. Shuffling the training data introduces randomness and promotes better generalization.
    # No shuffle for validation data: It is generally not necessary to shuffle the validation data because validation is meant to evaluate the model's performance on unseen data that is representative of the real-world scenarios. Shuffling the validation data could lead to inconsistent evaluation results between different validation iterations, making it harder to track the model's progress and compare performance.


    # 0. LOAD MODEL 
    # Option 1: FROM CHECKPOINT
    input_checkpoint_folder  = f'GraphRuns/{experiment_name}'
    input_checkpoint_path = f'{input_checkpoint_folder}/{input_checkpoint_name}'
    
    # Define network dimensions
    if config["model_type"] == "gnn":
        gnn_input_dim = dataset.targets[0].node_features.shape[1]
        gnn_hidden_dim = config["hidden_dim"]
        gnn_output_dim = config["output_dim"]

    elif config["model_type"] == "fingerprints":
        #     fingerprint_input_dim = preprocessed_targets[0].GetNumBits()
        fingerprint_input_dim = dataset.targets[0].size()[
            0
        ]  # len(preprocessed_targets[0].node_features)
        fingerprint_hidden_dim = config["hidden_dim"]
        fingerprint_output_dim = config["output_dim"]

    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model_type"] == "gnn":
        model = GNNModel(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
        ).to(device)
        model.double()

    elif config["model_type"] == "fingerprints":
        model = FingerprintModel(
            input_dim=fingerprint_input_dim,
            hidden_dim=fingerprint_hidden_dim,
            output_dim=fingerprint_output_dim,
        ).to(device)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    loss_fn = NTXentLoss(temperature=config["temperature"], device=device)

    checkpoint = torch.load(input_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    breakpoint()
            
    # # # OPTION 2: From pickle
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_fn = NTXentLoss(temperature=config["temperature"], device=device)
    # with open(f'{checkpoint_folder}/model_min_val.pkl', "rb") as f:
    #     model = pickle.load(f)

    
    # 3: EVALUATE MODEL
    model.eval()  # Set the model to evaluation mode
    
    # Loss on train
    train_loss = 0.0
    train_batches = 0
    
    with torch.no_grad():  # Disable gradient calculation during validation
        for train_batch_idx, train_batch_data in enumerate(train_data_loader):
            # Compute embeddings
            train_embeddings = compute_embeddings(
                device=device,
                model_type=config['model_type'],
                model=model,
                batch_data=train_batch_data,
            )
            # Compute loss
            train_batch_loss = loss_fn(train_embeddings)

            train_loss += train_batch_loss.item()
            train_batches += 1
        
        # Compute average loss
        average_train_loss = train_loss / train_batches
    
    print("Train loss: ", average_train_loss)

    # Loss on validation
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for val_batch_idx, val_batch_data in enumerate(val_data_loader):
            # Compute embeddings
            val_embeddings = compute_embeddings(
                device=device,
                model_type=config['model_type'],
                model=model,
                batch_data=val_batch_data,
            )
            # Compute loss
            val_batch_loss = loss_fn(val_embeddings)

            val_loss += val_batch_loss.item()
            val_batches += 1

        # Compute average loss
        average_val_loss = val_loss / val_batches
        
    print("Validation loss: ", average_val_loss)


