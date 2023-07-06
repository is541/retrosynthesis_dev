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

from tqdm.auto import tqdm
from paroutes import PaRoutesInventory, get_target_smiles
from embedding_model import (
    fingerprint_preprocess_input_react,
    gnn_preprocess_input_react,
    # CustomDataset,
    CustomDatasetReact,
    collate_fn_react,
    # SampleData,
    fingerprint_vect_from_smiles,
    compute_embeddings_react,
    GNNModel,
    FingerprintModel,
    # NTXentLoss,
    ContrastiveReact,
)
from paroutes import PaRoutesInventory
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import plotly.express as px
from rdkit import Chem
import deepchem as dc


# def num_heavy_atoms(mol):
#     return Chem.rdchem.Mol.GetNumAtoms(mol, onlyExplicit=True)


def compute_embedding_purch_mols(
    device, model_type, model, purch_featurizer, purch_fingerprints
):
    if model_type == "gnn":
        purch_embeddings = torch.stack(
            [
                model(
                    torch.tensor(purch_mol.node_features, dtype=torch.double).to(
                        device
                    ),
                    torch.tensor(purch_mol.edge_index, dtype=torch.long).to(device),
                )
                for purch_mol in purch_featurizer
            ],
            dim=0,
        )
    elif model_type == "fingerprints":
        purch_embeddings = torch.stack(
            [
                model(torch.tensor(fingerprint, dtype=torch.double).to(device))
                for fingerprint in purch_fingerprints
            ],
            dim=0,
        )
    else:
        raise NotImplementedError(f"Model type {model_type}")
    return purch_embeddings


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

    experiment_name = config["experiment_name"]
    checkpoint_folder = f"GraphRuns/{experiment_name}/"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_name = "checkpoint.pth"

    # if not args.load_from_preprocessed_data:
    # Save config in output folder
    with open(f"{checkpoint_folder}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Read first reaction data
    input_file_first_react = (
        # f'Runs/{config["run_id"]}/first_reaction_positive_negatives_not_purch.pickle'
        f'Runs/{config["run_id"]}/first_reaction_positive_negatives_all.pickle'
    )

    # First reaction data
    with open(input_file_first_react, "rb") as handle:
        targ_first_react_dict_original = pickle.load(handle)

    targ_first_react_dict = {}
    # targ_num_mols_each_negative = {}
    for target_smiles, values in targ_first_react_dict_original.items():
        positive_samples = values["positive_samples"]
        negative_samples = values["negative_samples"]

        flattened_positive_samples = [
            sample for sublist in positive_samples for sample in sublist
        ]

        flattened_negative_samples = [
            sample for sublist in negative_samples for sample in sublist
        ]
        num_mols_each_negative = [len(sublist) for sublist in negative_samples]

        targ_first_react_dict[target_smiles] = {
            "positive_samples": flattened_positive_samples,  # Only one positive sample (i.e. set of molecules), no need to keep track of how to split it
            "negative_samples": flattened_negative_samples,
            "num_mols_each_negative": torch.tensor(
                num_mols_each_negative, dtype=torch.double
            ),
            "cost_pos_react": torch.tensor(
                values["cost_pos_react"], dtype=torch.double
            ),
            "cost_neg_react": torch.tensor(
                values["cost_neg_react"], dtype=torch.double
            ),
        }
        # targ_num_mols_each_negative[target_smiles] = torch.tensor(num_mols_each_negative, dtype=torch.double)

    # # # Load distances data
    # # with open(input_file_distances, 'rb') as handle:
    # #     distances_dict = pickle.load(handle)

    # Inventory
    inventory = PaRoutesInventory(n=5)
    purch_smiles = [mol.smiles for mol in inventory.purchasable_mols()]
    # # len(purch_smiles)

    # def num_heavy_atoms(mol):
    #     return Chem.rdchem.Mol.GetNumAtoms(mol, onlyExplicit=True)

    # purch_mol_to_exclude = []
    # purch_nr_heavy_atoms = {}
    # for smiles in purch_smiles:
    #     nr_heavy_atoms = num_heavy_atoms(Chem.MolFromSmiles(smiles))
    #     if nr_heavy_atoms < 2:
    #         purch_mol_to_exclude = purch_mol_to_exclude + [smiles]
    #     purch_nr_heavy_atoms[smiles] = nr_heavy_atoms

    # if config["run_id"] == "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7":
    #     all_targets = get_target_smiles(n=5)
    # elif config["run_id"] == "Guacamol_combined":
    #     with open("Data/Guacamol/guacamol_v1_test_10ksample.txt", "r") as f:
    #         all_targets = [line.strip() for line in f.readlines()]

    # targ_route_not_in_route_dict = {}
    # for target in all_targets:
    #     targ_route_not_in_route_dict[target] = {}

    #     target_routes_dict = targ_routes_dict.get(target, "Target_Not_Solved")

    #     if target_routes_dict == "Target_Not_Solved":
    #         purch_in_route = []
    #     else:
    #         target_route_df = target_routes_dict["route_1"]
    #         purch_in_route = list(
    #             target_route_df.loc[target_route_df["label"] != "Target", "smiles"]
    #         )
    #     #         purch_in_route = [smiles for smiles in purch_in_route if smiles in purch_smiles]
    #     purch_not_in_route = [
    #         purch_smile
    #         for purch_smile in purch_smiles
    #         if purch_smile not in purch_in_route
    #     ]
    #     random.seed(config["seed"])

    #     if config["neg_sampling"] == "uniform":
    #         purch_not_in_route_sample = random.sample(
    #             purch_not_in_route, config["not_in_route_sample_size"]
    #         )
    #     elif config["neg_sampling"] == "...":
    #         pass
    #     else:
    #         raise NotImplementedError(f'{config["neg_sampling"]}')

    #     # Filter out molecules with only one atom (problems with featurizer)
    #     purch_in_route = [
    #         smiles for smiles in purch_in_route if smiles not in purch_mol_to_exclude
    #     ]
    #     purch_not_in_route_sample = [
    #         smiles
    #         for smiles in purch_not_in_route_sample
    #         if smiles not in purch_mol_to_exclude
    #     ]

    #     targ_route_not_in_route_dict[target]["positive_samples"] = purch_in_route
    #     targ_route_not_in_route_dict[target][
    #         "negative_samples"
    #     ] = purch_not_in_route_sample

    # Get a random sample of keys from targ_routes_dict
    if config["nr_sample_targets"] != -1:
        sample_targets = random.sample(
            list(targ_first_react_dict.keys()), config["nr_sample_targets"]
        )
        targ_first_react_dict_sample = {
            target: targ_first_react_dict[target] for target in sample_targets
        }
    else:
        targ_first_react_dict_sample = targ_first_react_dict
    # Create targ_routes_dict_sample with the sampled keys and their corresponding values

    input_data = targ_first_react_dict_sample

    if config["model_type"] == "gnn":
        featurizer = dc.feat.MolGraphConvFeaturizer()

        purch_mols = [Chem.MolFromSmiles(smiles) for smiles in purch_smiles]
        purch_featurizer = featurizer.featurize(purch_mols)
        purch_featurizer_dict = dict(zip(purch_smiles, purch_featurizer))
        with open(f"{checkpoint_folder}/purch_featurizer_dict.pickle", "wb") as handle:
            pickle.dump(purch_featurizer_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fingerprint_num_atoms_dict = None

        dataset = gnn_preprocess_input_react(
            input_data=input_data,
            featurizer=featurizer,
            featurizer_dict=None,
        )

    elif config["model_type"] == "fingerprints":
        purch_fingerprints = list(map(fingerprint_vect_from_smiles, purch_smiles))
        purch_fingerprints_dict = dict(zip(purch_smiles, purch_fingerprints))
        with open(
            f"{checkpoint_folder}/purch_fingerprints_dict.pickle", "wb"
        ) as handle:
            pickle.dump(
                purch_fingerprints_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        # # Also save dict to retrieve number of atoms from fingerprints
        # fingerprint_num_atoms_dict = {
        #     torch.tensor(fp, dtype=torch.double): purch_nr_heavy_atoms[smiles]
        #     for smiles, fp in purch_fingerprints_dict.items()
        # }
        # with open(
        #     f"{checkpoint_folder}/fingerprint_num_atoms_dict.pickle", "wb"
        # ) as handle:
        #     pickle.dump(
        #         fingerprint_num_atoms_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
        #     )

        dataset = fingerprint_preprocess_input_react(
            input_data=input_data,
            fingerprints_dict=None,
        )
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    # Train validation split
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
        collate_fn=collate_fn_react,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=config["val_shuffle"],
        collate_fn=collate_fn_react,
    )

    # Batch size: The batch size determines the number of samples processed in each iteration during training or validation. In most cases, it is common to use the same batch size for both training and validation to maintain consistency. However, there are situations where you might choose a different batch size for validation. For instance, if memory constraints are more relaxed during validation, you can use a larger batch size to speed up evaluation.
    # Shuffle training data: Shuffling the training data before each epoch is beneficial because it helps the model see the data in different orders, reducing the risk of the model learning patterns specific to the order of the data. Shuffling the training data introduces randomness and promotes better generalization.
    # No shuffle for validation data: It is generally not necessary to shuffle the validation data because validation is meant to evaluate the model's performance on unseen data that is representative of the real-world scenarios. Shuffling the validation data could lead to inconsistent evaluation results between different validation iterations, making it harder to track the model's progress and compare performance.

    # Define network dimensions
    if config["model_type"] == "gnn":
        gnn_input_dim = dataset.targets[0].node_features.shape[1]
        gnn_hidden_dim = config["hidden_dim"]
        gnn_output_dim = config["output_dim"]

        with open(f"{checkpoint_folder}/input_dim.pickle", "wb") as f:
            pickle.dump({"input_dim": gnn_input_dim}, f)

    elif config["model_type"] == "fingerprints":
        #     fingerprint_input_dim = preprocessed_targets[0].GetNumBits()
        # print(dataset.targets[0])
        fingerprint_input_dim = dataset.targets[0].size()[
            0
        ]  # len(preprocessed_targets[0].node_features)
        fingerprint_hidden_dim = config["hidden_dim"]
        fingerprint_output_dim = config["output_dim"]

        with open(f"{checkpoint_folder}/input_dim.pickle", "wb") as f:
            pickle.dump({"input_dim": fingerprint_input_dim}, f)

    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    # Step 3: Set up the training loop for the GNN model
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

    loss_fn = ContrastiveReact(temperature=config["temperature"], device=device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    num_epochs = config["num_epochs"]

    load_from_checkpoint = False

    # STEP 5: Train loop
    # Check if a checkpoint exists and load the model state and optimizer state if available
    if load_from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    # Create a SummaryWriter for TensorBoard logging
    log_dir = (
        f"{checkpoint_folder}/logs"  # Specify the directory to store TensorBoard logs
    )
    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")
    best_model = None

    epoch_loss = pd.DataFrame(columns=["Epoch", "TrainLoss", "ValLoss"])
    for epoch in tqdm(range(start_epoch, num_epochs)):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch_data in enumerate(train_data_loader):
            optimizer.zero_grad()

            # Compute embeddings for target and positive samples
            embeddings_dataset = compute_embeddings_react(
                device=device,
                model_type=config["model_type"],
                model=model,
                batch_data=batch_data,
            )

            # Compute embeddings for purchasable molecules
            purch_embeddings = compute_embedding_purch_mols(
                device=device,
                model_type=config["model_type"],
                model=model,
                purch_featurizer=purch_featurizer if "purch_featurizer" in locals() else None,
                purch_fingerprints=purch_fingerprints if "purch_fingerprints" in locals() else None,
            )

            # Compute loss
            loss = loss_fn(embeddings_dataset, purch_embeddings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track total loss
            train_loss += loss.item()
            train_batches += 1

        # VALIDATION
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():  # Disable gradient calculation during validation
            for val_batch_idx, val_batch_data in enumerate(val_data_loader):
                # Compute embeddings
                val_embeddings = compute_embeddings_react(
                    device=device,
                    model_type=config["model_type"],
                    model=model,
                    batch_data=val_batch_data,
                )

                # Compute loss
                val_batch_loss = loss_fn(val_embeddings, purch_embeddings)

                val_loss += val_batch_loss.item()
                val_batches += 1

        # METRICS
        # - TRAIN
        # Compute average loss for the epoch
        average_train_loss = train_loss / train_batches

        # Log the loss to TensorBoard
        writer.add_scalar("Loss/train", average_train_loss, epoch + 1)

        # - VALIDATION
        average_val_loss = val_loss / val_batches

        # Log the loss to TensorBoard
        writer.add_scalar("Loss/val", average_val_loss, epoch + 1)

        new_row = pd.DataFrame(
            {
                "Epoch": [epoch],
                "TrainLoss": [average_train_loss],
                "ValLoss": [average_val_loss],
            }
        )
        epoch_loss = pd.concat([epoch_loss, new_row], axis=0)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model = model

        if (epoch % 10 == 0) | (epoch == num_epochs - 1):
            print(
                f"{config['model_type']} Model - Epoch {epoch+1}/{num_epochs}, TrainLoss: {average_train_loss}, ValLoss: {average_val_loss}"
            )

            # Save the model and optimizer state as a checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint_path = f"{checkpoint_folder}/epoch_{epoch+1}_{checkpoint_name}"  # Specify the checkpoint file path
            torch.save(checkpoint, checkpoint_path)

            #         loss_df = pd.DataFrame({'Epoch': range(len(epoch_loss)), 'TrainLoss': epoch_loss})
            epoch_loss.to_csv(f"{checkpoint_folder}/train_val_loss.csv", index=False)

            # Save the best model as a pickle
            best_model_path = (
                f"{checkpoint_folder}/model_min_val.pkl"  #'path/to/best_model.pkl'
            )

            with open(best_model_path, "wb") as f:
                pickle.dump(best_model, f)

    # Close the SummaryWriter
    writer.close()

    # STEP 6: Plot

    # fig = px.line(x=epoch_loss['Epoch'], y=epoch_loss['TrainLoss'], title="Train loss")
    # fig.update_layout(width=1000, height=600, showlegend=False)
    # fig.write_image(f"{checkpoint_folder}/Train_loss.pdf")
    # fig.show()

    # Create a new figure with two lines
    fig = px.line()

    # Add the TrainLoss line to the figure
    fig.add_scatter(x=epoch_loss["Epoch"], y=epoch_loss["TrainLoss"], name="Train Loss")

    # Add the ValLoss line to the figure
    fig.add_scatter(
        x=epoch_loss["Epoch"], y=epoch_loss["ValLoss"], name="Validation Loss"
    )

    # Set the title of the figure
    fig.update_layout(title="Train and Validation Loss")

    # Set the layout size and show the legend
    fig.update_layout(width=1000, height=600, showlegend=True)

    # Save the figure as a PDF file
    fig.write_image(f"{checkpoint_folder}/Train_and_Val_loss.pdf")
